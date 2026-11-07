"""相位流形步态模块三段训练：PAE（train GT 腿部特征）→ 逐帧 GT 相位 → G（GT 相位 + GT root）→ F（G 冻结）。

设计与预登记：docs/ai/context/20261107-095645-ntu2p-phase-manifold-gait-design-and-preregistration.md。
只读 train split；G 以 GT 相位训练，因此学到的是"给定相位时腿的形状"而非对相位求平均；
F 不直接接腿部 L2（避免为对冲相位不确定而缩小振幅），只用相位/频率/振幅监督与经冻结 G 的自洽支撑项。
"""

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch
from torch.nn import functional as F

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache
from model.ntu2p_phase_gait import (
    FEAT_DIM, GEN_JOINTS, L_HIP, R_HIP, PeriodicAutoencoder, PhaseGaitHead, bone_lengths, dct_basis, extend_back, leg_directions, leg_features,
    pelvis_height, phase_vector,
)
from utils.fixseed import fixseed
from utils.ntu2p_canonical import estimate_up
from utils.ntu2p_kinematic_projection import foot_contact_mask, foot_horizontal_velocity, foot_skate_loss, observed_ground

PARAM_KEYS = ("f", "a", "b", "psi")


def log(handle, record):
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def sequence_features(cache):
    """逐序列（序列级 up）算腿部特征，拼成与 cache.xyz 同序的 [F,2,21]。"""
    out = []
    with torch.no_grad():
        for index in range(len(cache)):
            seq = cache.sequence(index).unsqueeze(0)
            feat, _ = leg_features(seq, estimate_up(seq))
            out.append(feat[0])
    return torch.cat(out, dim=0)


def padded_index(cache, half):
    """每帧中心窗口在"两端各复制 half 帧"的展开下的起点；返回 [F] 窗口首帧的全局帧号矩阵 [F,W]。"""
    rows = []
    for index in range(len(cache)):
        start, length = int(cache.offsets_cpu[index]), int(cache.lengths_cpu[index])
        centers = torch.arange(length).unsqueeze(1)
        local = (centers + torch.arange(-half, half + 1).unsqueeze(0)).clamp(0, length - 1)
        rows.append(local + start)
    return torch.cat(rows, dim=0)


def train_pae(feats, index, opts, device, handle):
    pae = PeriodicAutoencoder(channels=opts.channels, window=opts.window).to(device)
    optim = torch.optim.Adam(pae.parameters(), lr=opts.lr)
    gen = torch.Generator().manual_seed(opts.seed)
    total = index.shape[0]
    for step in range(1, opts.pae_steps + 1):
        frame = torch.randint(total, (opts.batch,), generator=gen)
        person = torch.randint(2, (opts.batch,), generator=gen)
        x = feats[index[frame].to(device), person.to(device)[:, None]].transpose(1, 2)
        recon, _ = pae(x)
        loss = F.mse_loss(recon, x)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 500 == 0 or step == 1:
            log(handle, OrderedDict([("stage", "pae"), ("step", step), ("recon_mse", float(loss))]))
    return pae.eval()


def phase_params(pae, feats, index, device, chunk=4096):
    """全部帧（两人）的中心窗口 PAE 参数 -> OrderedDict(k -> [F,2,K])。"""
    out = OrderedDict((k, []) for k in PARAM_KEYS)
    with torch.no_grad():
        for begin in range(0, index.shape[0], chunk):
            rows = index[begin : begin + chunk].to(device)
            x = feats[rows]  # [n,W,2,21]
            n = x.shape[0]
            x = x.permute(0, 2, 3, 1).reshape(n * 2, FEAT_DIM, -1)
            p = pae.params(x)
            for k in PARAM_KEYS:
                out[k].append(p[k].view(n, 2, -1))
    return OrderedDict((k, torch.cat(v, dim=0)) for k, v in out.items())


def sample_windows(cache, batch, window_len, gen):
    """随机序列、随机起点的长度 window_len 窗口：返回 xyz [B,W,2,55,3]、全局帧号 [B,W]、动作 [B]。"""
    seq = torch.randint(len(cache), (batch,), generator=gen)
    room = (cache.lengths_cpu[seq] - window_len + 1).clamp_min(1)
    start = (torch.rand(batch, generator=gen) * room.float()).long()
    frames = (cache.offsets_cpu[seq] + start).unsqueeze(1) + torch.arange(window_len).unsqueeze(0)
    frames = frames.to(cache.device)
    return cache.xyz[frames], frames, cache.actions[seq.to(cache.device)]


def joints_shape(xyz):
    return (xyz.shape[0], xyz.shape[1], 2, len(GEN_JOINTS))


def gather(params, frames):
    return OrderedDict((k, v[frames]) for k, v in params.items())


def window_inputs(xyz, obs_len):
    obs, target = xyz[:, :obs_len], xyz[:, obs_len:]
    up = estimate_up(obs)
    feat, frames = leg_features(xyz, up)
    obs_dirs = leg_directions(xyz[:, obs_len - 1 : obs_len], frames[:, obs_len - 1 : obs_len])[:, 0]
    return obs, target, up, feat, frames, pelvis_height(xyz, obs, up), obs_dirs


FOOT_SLOTS = (1, 2, 4, 5)  # GEN_JOINTS 中踝、脚掌的位置


def penetration(joints, obs, up):
    """生成的踝/脚掌低于观测地面部分的平方均值（m²）。"""
    height = (joints[..., list(FOOT_SLOTS), :] * up[:, None, None, None, :]).sum(-1) - observed_ground(obs, up)[:, None, :, None]
    return F.relu(-height).pow(2).mean()


def with_legs(target, joints):
    out = target.clone()
    out[..., list(GEN_JOINTS), :] = joints
    return out


def train_decoder(head, cache, params, opts, device, handle):
    decoder = head.decoder
    optim = torch.optim.Adam(decoder.parameters(), lr=opts.lr)
    gen = torch.Generator().manual_seed(opts.seed + 1)
    window_len = opts.obs_len + opts.pred_len
    for step in range(1, opts.g_steps + 1):
        xyz, frames_idx, action = sample_windows(cache, opts.batch_g, window_len, gen)
        obs, target, up, feat, frames, height, obs_dirs = window_inputs(xyz, opts.obs_len)
        lengths = bone_lengths(obs[:, -1])
        hips = xyz[..., [L_HIP, R_HIP], :]
        weight = torch.ones(joints_shape(xyz), device=device)
        if opts.ground_aware and opts.hip_drop_prob > 0:
            # 推理时主体给的骨盆可能比真实低（穿地帧上平均低 9 cm）：降低未来帧的髋、脚目标不动，教解码器屈膝落地。
            batch = xyz.shape[0]
            dropped = (torch.rand(batch, 1, 2, 1, generator=gen) < opts.hip_drop_prob).float().to(device)
            drop = torch.rand(batch, 1, 2, 1, generator=gen).to(device) * opts.hip_drop_max * dropped
            ramp = ((torch.arange(xyz.shape[1], device=device) - opts.obs_len + 1).clamp(0, 5).float() / 5).view(1, -1, 1, 1)
            offset = drop * ramp
            hips = hips - offset[..., None] * up[:, None, None, None, :]
            height = height - offset
            weight[..., 0] = 1 - dropped[..., 0]
            weight[..., 3] = 1 - dropped[..., 0]
        joints, logits = head.legs(gather(params, frames_idx), feat[..., -3:], hips, lengths, frames, action, height, obs_dirs)
        err = (joints - xyz[..., list(GEN_JOINTS), :]).norm(dim=-1)
        dist = (weight * err).sum() / weight.sum()
        mask = foot_contact_mask(target, obs)
        pred_target = with_legs(target, joints[:, opts.obs_len :])
        skate = foot_skate_loss(pred_target, target, obs, up, mask)
        bce = F.binary_cross_entropy_with_logits(logits[:, opts.obs_len :], mask)
        pen = penetration(joints[:, opts.obs_len :], obs, up)
        loss = dist + opts.skate_weight * skate + opts.contact_weight * bce + opts.penetration_weight * pen
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 500 == 0 or step == 1:
            log(handle, OrderedDict([("stage", "G"), ("step", step), ("leg_dist", float(dist)), ("skate", float(skate)), ("bce", float(bce)),
                                     ("penetration", float(pen))]))
    for p in decoder.parameters():
        p.requires_grad_(False)


def circular(pred, target, weight):
    return (weight * (1 - torch.cos(2 * math.pi * (pred - target)))).sum() / weight.sum().clamp_min(1e-8)


def train_forecaster(head, cache, params, scales, opts, device, handle):
    forecaster = head.forecaster
    optim = torch.optim.Adam(forecaster.parameters(), lr=opts.lr)
    gen = torch.Generator().manual_seed(opts.seed + 2)
    noise_gen = torch.Generator(device=device).manual_seed(opts.seed + 3)
    basis = dct_basis(opts.pred_len, max(opts.vel_lowpass_order, 1), device=device)
    window_len = opts.obs_len + opts.pred_len
    for step in range(1, opts.f_steps + 1):
        xyz, frames_idx, action = sample_windows(cache, opts.batch_f, window_len, gen)
        obs, target, up, feat, frames, height, obs_dirs = window_inputs(xyz, opts.obs_len)
        velocity = feat[..., -3:]
        future_vel = velocity[:, opts.obs_len :]
        if opts.vel_lowpass_order > 0:
            # 主体 root 是 8 阶 DCT 带限的确定性输出，走向不确定时偏慢：一半样本用低通 + 随机收缩后的速度作输入。
            low = torch.einsum("tk,sk,bspc->btpc", basis, basis, future_vel)
            shrink = 0.6 + 0.4 * torch.rand(future_vel.shape[0], 1, 1, 1, generator=gen).to(device)
            use = (torch.rand(future_vel.shape[0], 1, 1, 1, generator=gen) < 0.5).float().to(device)
            future_vel = use * low * shrink + (1 - use) * future_vel
        # 推理时未来骨盆速度来自主体模型（比 GT 平滑、有误差），训练时加相对 + 绝对噪声模拟。
        noise = torch.randn(future_vel.shape, generator=noise_gen, device=device)
        noisy = future_vel + noise * (opts.vel_noise_rel * future_vel.norm(dim=-1, keepdim=True) + opts.vel_noise_abs)
        obs_feat = (feat[:, : opts.obs_len] - head.feat_mean) / head.feat_std
        pred, psi0 = forecaster(obs_feat, noisy, action)
        gt = gather(params, frames_idx)
        future = OrderedDict((k, v[:, opts.obs_len :]) for k, v in gt.items())
        weight = future["a"]
        phase = circular(pred["psi"], future["psi"], weight) + circular(psi0, gt["psi"][:, opts.obs_len - 1], gt["a"][:, opts.obs_len - 1])
        mags = sum(((pred[k] - future[k]) / scales[k]).pow(2).mean() for k in ("f", "a", "b"))
        full = extend_back(pred, psi0, opts.obs_len)
        lengths = bone_lengths(obs[:, -1])
        joints, logits = head.legs(full, velocity, xyz[..., [L_HIP, R_HIP], :], lengths, frames, action, height, obs_dirs)
        pred_target = with_legs(target, joints[:, opts.obs_len :])
        prob = torch.sigmoid(logits[:, opts.obs_len :]).detach()
        speed_sq = foot_horizontal_velocity(pred_target, obs, up).pow(2).sum(-1)
        self_skate = (prob * speed_sq).sum() / prob.sum().clamp_min(1.0)
        # 分析—合成：反推到观测帧的相位须复现观测到的腿，既校准当前相位，也消除观测/预测衔接处的跳变。
        recon = (joints[:, : opts.obs_len] - obs[..., list(GEN_JOINTS), :]).norm(dim=-1).mean()
        # v4：未来腿关节 L2 经冻结 G 回传，只能改相位/频率/振幅/偏置——时机不确定时少迈、迈小，是精度与自然度的显式权衡。
        leg_l2 = (joints[:, opts.obs_len :] - target[..., list(GEN_JOINTS), :]).norm(dim=-1).mean()
        loss = phase + mags + opts.self_skate_weight * self_skate + opts.obs_recon_weight * recon + opts.leg_l2_weight * leg_l2
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 500 == 0 or step == 1:
            log(handle, OrderedDict([("stage", "F"), ("step", step), ("phase", float(phase)), ("mags", float(mags)), ("self_skate", float(self_skate)),
                                     ("obs_recon", float(recon)), ("leg_l2", float(leg_l2))]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--cache_dir", default="results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval")
    parser.add_argument("--manifest_path", default="results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--window", type=int, default=41)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--batch_g", type=int, default=128)
    parser.add_argument("--batch_f", type=int, default=128)
    parser.add_argument("--pae_steps", type=int, default=4000)
    parser.add_argument("--g_steps", type=int, default=6000)
    parser.add_argument("--f_steps", type=int, default=6000)
    parser.add_argument("--skate_weight", type=float, default=0.2)
    parser.add_argument("--contact_weight", type=float, default=0.05)
    parser.add_argument("--self_skate_weight", type=float, default=0.2)
    parser.add_argument("--vel_noise_rel", type=float, default=0.15)
    parser.add_argument("--vel_noise_abs", type=float, default=0.03)
    parser.add_argument("--ground_aware", action="store_true", help="v2：解码器输入骨盆离观测地面高度，配髋下降增强与穿地损失")
    parser.add_argument("--hip_drop_prob", type=float, default=0.5)
    parser.add_argument("--hip_drop_max", type=float, default=0.12)
    parser.add_argument("--penetration_weight", type=float, default=0.0)
    parser.add_argument("--obs_recon_weight", type=float, default=0.0)
    parser.add_argument("--obs_pose_input", action="store_true", help="v3：解码器以观测末帧骨方向为条件")
    parser.add_argument("--anchor_tau", type=float, default=0.0, help="v3：观测锚定衰减帧数，0 关闭")
    parser.add_argument("--vel_lowpass_order", type=int, default=0, help="v3：一半样本把速度输入截断到该 DCT 阶数并随机收缩，0 关闭")
    parser.add_argument("--leg_l2_weight", type=float, default=0.0, help="v4：相位预测器的未来腿关节 L2（经冻结解码器）")
    opts = parser.parse_args()
    if not opts.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("步态模块训练必须显式使用可用的 CUDA 设备")
    fixseed(opts.seed)
    device = torch.device(opts.device)
    save = Path(opts.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    handle = open(save / "train_log.jsonl", "a")
    cache = NTU2PXYZSeqCache.load(opts.cache_dir, "train", manifest_path=opts.manifest_path, device=device)
    feats = sequence_features(cache)
    mean, std = feats.reshape(-1, FEAT_DIM).mean(0), feats.reshape(-1, FEAT_DIM).std(0).clamp_min(1e-4)
    normed = (feats - mean) / std
    index = padded_index(cache, opts.window // 2)
    pae = train_pae(normed, index, opts, device, handle)
    params = phase_params(pae, normed, index, device)
    config = OrderedDict([("channels", opts.channels), ("obs_len", opts.obs_len), ("pred_len", opts.pred_len), ("order", opts.order),
                          ("ground_aware", bool(opts.ground_aware)), ("obs_pose_input", bool(opts.obs_pose_input)), ("anchor_tau", float(opts.anchor_tau))])
    head = PhaseGaitHead(config).to(device)
    vectors = phase_vector(params).reshape(-1, 4 * opts.channels)
    with torch.no_grad():
        head.feat_mean.copy_(mean)
        head.feat_std.copy_(std)
        head.phase_mean.copy_(vectors.mean(0))
        head.phase_std.copy_(vectors.std(0).clamp_min(1e-4))
    scales = OrderedDict((k, params[k].std().clamp_min(1e-4)) for k in ("f", "a", "b"))
    log(handle, OrderedDict([("stage", "phase_stats")] + [(k, float(params[k].mean())) for k in PARAM_KEYS] + [("scale_" + k, float(v)) for k, v in scales.items()]))
    train_decoder(head, cache, params, opts, device, handle)
    train_forecaster(head, cache, params, scales, opts, device, handle)
    torch.save(OrderedDict([("config", config), ("head", head.state_dict()), ("pae", pae.state_dict()),
                            ("pae_config", OrderedDict([("channels", opts.channels), ("window", opts.window)])), ("args", vars(opts))]),
               save / "phase_gait.pt")
    handle.close()


if __name__ == "__main__":
    main()
