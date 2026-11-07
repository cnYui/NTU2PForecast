"""相位流形步态模块第 0 阶段：S0-1（GT 相位 + GT root）、S0-2（预测相位 + GT root）诊断，S0-3（BR:ode0 + 步态模块）完整评价与判定。

预登记：docs/ai/context/20261107-095645-ntu2p-phase-manifold-gait-design-and-preregistration.md。
S0-3 与 BR:ode0（不加任何后处理）比，mpjpe 容许 1.005×；其余护栏与落脚各轮相同。
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, eval_windows
from model.ntu2p_phase_gait import (
    GEN_JOINTS, L_HIP, R_HIP, PeriodicAutoencoder, bone_lengths, extend_back, leg_directions, leg_features, load_gait_head, pelvis_height,
)
from scripts.run_ntu2p_bandflow import eval_readout
from scripts.run_ntu2p_footloss import decide, gait_level
from scripts.run_ntu2p_footloss_sample import seed_row
from scripts.run_ntu2p_jitflow import BASE, PROTOCOLS, say
from train.train_ntu2p_phase_gait import padded_index, phase_params, sequence_features
from utils.ntu2p_canonical import estimate_up
from utils.ntu2p_naturalness import aggregate_stats, compute_naturalness_stats

MPJPE_TOL = 1.005
GAIT_KEYS = ("step_count", "slide_frame_ratio", "stance_fraction", "lr_forward_vel_corr", "skate_gt_contact_speed_walk")


def summarize(pred, target, obs):
    # 自然度统计在 CPU 上算：整批 247 窗的复制填充在 GPU 上超出 kernel 配置上限。
    pred, target, obs = pred.cpu(), target.cpu(), obs.cpu()
    stats = aggregate_stats(compute_naturalness_stats(pred.double(), target.double(), obs.double()))
    row = OrderedDict((k, float(stats[k])) for k in GAIT_KEYS)
    row["mpjpe"] = float((pred.double() - target.double()).norm(dim=-1).mean())
    row["leg_err"] = float((pred[..., list(GEN_JOINTS), :].double() - target[..., list(GEN_JOINTS), :].double()).norm(dim=-1).mean())
    return row


def diagnostics(checkpoint, protocol, device, br_arrays):
    """S0-1 / S0-2：只换腿，root/上半身用 GT，看相位流形与相位预测本身。"""
    state = torch.load(str(checkpoint), map_location="cpu")
    head = load_gait_head(str(checkpoint), device)
    pae = PeriodicAutoencoder(**state["pae_config"]).to(device).eval()
    pae.load_state_dict(state["pae"])
    cache = NTU2PXYZSeqCache.load(protocol["cache"], "val", manifest_path=protocol["manifest"], device=device)
    windows = eval_windows(cache)
    obs, target, action = windows["obs_xyz"].to(device), windows["target_xyz"].to(device), windows["action"].to(device)
    with torch.no_grad():
        feats = (sequence_features(cache) - head.feat_mean) / head.feat_std
        params = phase_params(pae, feats, padded_index(cache, int(state["pae_config"]["window"]) // 2), device)
        obs_len = int(obs.shape[1])
        frames_idx = (cache.offsets.cpu() + windows["start"]).unsqueeze(1) + torch.arange(obs_len + target.shape[1]).unsqueeze(0)
        gt = OrderedDict((k, v[frames_idx.to(device)]) for k, v in params.items())
        full = torch.cat((obs, target), dim=1)
        up = estimate_up(obs)
        feat, frames = leg_features(full, up)
        velocity, hips, lengths = feat[..., -3:], full[..., [L_HIP, R_HIP], :], bone_lengths(obs[:, -1])
        height = pelvis_height(full, obs, up)
        obs_dirs = leg_directions(full[:, obs_len - 1 : obs_len], frames[:, obs_len - 1 : obs_len])[:, 0]
        rows = OrderedDict()
        rows["GT"] = summarize(target, target, obs)
        oracle, _ = head.legs(gt, velocity, hips, lengths, frames, action, height, obs_dirs)
        rows["S0-1 GT 相位 + GT root"] = summarize(_replace(target, oracle[:, obs_len:]), target, obs)
        obs_feat = (feat[:, :obs_len] - head.feat_mean) / head.feat_std
        pred, psi0 = head.forecaster(obs_feat, velocity[:, obs_len:], action)
        predicted, _ = head.legs(extend_back(pred, psi0, obs_len), velocity, hips, lengths, frames, action, height, obs_dirs)
        rows["S0-2 预测相位 + GT root"] = summarize(_replace(target, predicted[:, obs_len:]), target, obs)
        br = torch.load(str(br_arrays))
        if not torch.equal(br["target_xyz"].to(device), target):
            raise ValueError("BR 导出数组与 val 窗口不一致")
        br_pred = br["methods"]["model"].to(device)
        rows["BR:ode0 腿（参照）"] = summarize(br_pred, target, obs)
        rows["BR:ode0 的腿接到 GT root（参照）"] = summarize(_replace(target, br_pred[..., list(GEN_JOINTS), :] - br_pred[..., :1, :] + target[..., :1, :]), target, obs)
    return rows


def _replace(target, joints):
    out = target.clone()
    out[..., list(GEN_JOINTS), :] = joints
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--version", default="v1", choices=("v1", "v2", "v3", "v4a", "v4b"))
    opts = parser.parse_args()
    tag = "" if opts.version == "v1" else "_" + opts.version
    protocol = PROTOCOLS["subjval"]
    sub = BASE / "ntu2p_jitflow" / "subjval"
    out = sub / "phase_gait{}_s{}".format(tag, opts.seed)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path("save/forecasting/ntu120_label/ntu2p_phase_gait{}_subjval_s{}/phase_gait.pt".format(tag, opts.seed))
    br_ckpt = Path("save/forecasting/ntu120_label/ntu2p_bandflow_subjval_BR_s{}_10000/ema/model000010000.pt".format(opts.seed))
    br_dir = sub / "footloss_s{}".format(opts.seed)
    br_dir.mkdir(parents=True, exist_ok=True)
    # 参照 BR:ode0 不加任何后处理，与 footloss_s0 同一命令；其它 seed 首次用到时补评。
    eval_readout(br_ckpt, br_dir / "BR_ode0.json", protocol, br_dir / "BR.log", "sample", 0.0, export=br_dir / "BR_ode0_arrays.pt")
    say("S0-1/S0-2 诊断")
    diag = diagnostics(checkpoint, protocol, torch.device(opts.device), br_dir / "BR_ode0_arrays.pt")
    say("S0-3 完整评价")
    eval_readout(br_ckpt, out / "PG_ode0.json", protocol, out / "PG.log", "sample", 0.0, export=out / "PG_ode0_arrays.pt",
                 extra=["--gait_head", str(checkpoint)])
    rows = OrderedDict([("BR:ode0", seed_row(br_dir, "BR", "ode0")), ("PGait", seed_row(out, "PG", "ode0"))])
    guards, level = decide(rows["PGait"], rows["BR:ode0"], MPJPE_TOL)
    if level == "无效（护栏未过）" and gait_level(rows["PGait"], rows["BR:ode0"]):
        level += "；步态条件{}".format("达标（" + gait_level(rows["PGait"], rows["BR:ode0"]) + "级）")
    summary = OrderedDict([("diagnostics", diag), ("rows", rows), ("guards", guards), ("level", level)])
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    keys = list(next(iter(diag.values())))
    lines = ["# 相位流形步态模块 {} 第 0 阶段（H-val seed{}）".format(opts.version, opts.seed), "", "## 诊断（只换腿）", "",
             "|行|" + "|".join(keys) + "|", "|---|" + "---:|" * len(keys)]
    lines += ["|{}|{}|".format(name, "|".join("{:.4f}".format(v) for v in row.values())) for name, row in diag.items()]
    keys = list(rows["BR:ode0"])
    lines += ["", "## S0-3 完整评价", "", "|行|" + "|".join(keys) + "|", "|---|" + "---:|" * len(keys)]
    lines += ["|{}|{}|".format(name, "|".join("{:.4f}".format(v) for v in row.values())) for name, row in rows.items()]
    lines += ["", "判定：" + level] + ["- 护栏 {}：{}".format(k, "过" if v else "未过") for k, v in guards.items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
