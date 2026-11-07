"""相位流形步态模块（PGait）：周期自编码器 PAE + 步态解码器 G + 相位预测器 F。

设计与预登记：docs/ai/context/20261107-095645-ntu2p-phase-manifold-gait-design-and-preregistration.md。
确定性预测在姿态坐标上对不同步相位求平均会把交替步子抹平；这里把"何时迈"（相位/频率/振幅，平均不相消）
与"给定相位时腿的形状"（G 以 GT 相位训练，没有平均）分开学，推理时 F 给相位、G 解码、FK 按骨长拼回。
"""

import math
from collections import OrderedDict

import torch
import torch.fft
from torch import nn
from torch.nn import functional as F

from model.forecasting_ntu2p_residual_xyz import NUM_ACTIONS
from utils.ntu2p_canonical import horizontal
from utils.ntu2p_kinematic_projection import observed_ground
from utils.ntu2p_naturalness import NTU2P_FPS

PELVIS, L_HIP, R_HIP = 0, 1, 2
# 每条腿：髋 → 膝 → 踝 → 脚掌；只生成膝/踝/脚掌，髋随主体。
CHAINS = ((1, 4, 7, 10), (2, 5, 8, 11))
GEN_JOINTS = (4, 7, 10, 5, 8, 11)
FEAT_DIM = len(GEN_JOINTS) * 3 + 3
NUM_BONES = 6
CONTACT_JOINTS = (7, 8, 10, 11)
# 站立时骨盆离脚底约 0.9 m；减去后作为解码器输入，量级与其它标准化特征相近。
PELVIS_HEIGHT_REF = 0.9


def _unit(value, eps=1e-8):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)


def _smooth_time(value, dim):
    """5 点二项平滑（边缘复制），只用于朝向：髋连线逐帧有拟合抖动。"""
    kernel = value.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    moved = value.transpose(dim, -1)
    shape = moved.shape
    flat = moved.reshape(-1, 1, shape[-1])
    padded = F.pad(flat, (2, 2), mode="replicate")
    return F.conv1d(padded, kernel.view(1, 1, 5)).reshape(shape).transpose(dim, -1)


def heading_frames(xyz, up):
    """xyz [B,T,P,55,3]、up [B,3] -> R [B,T,P,3,3]，列为 (左向, 前向, 上)；本人朝向系，随时间平滑。"""
    lateral = horizontal(xyz[..., L_HIP, :] - xyz[..., R_HIP, :], up)
    lateral = _unit(_smooth_time(lateral, 1))
    up_b = up[:, None, None, :].expand_as(lateral)
    forward = _unit(torch.cross(up_b, lateral, dim=-1))
    return torch.stack((lateral, forward, up_b), dim=-1)


def pelvis_velocity(xyz):
    """[B,T,P,3] 骨盆速度（m/s），中心差分、两端单侧。"""
    pelvis = xyz[..., PELVIS, :]
    padded = torch.cat((2 * pelvis[:, :1] - pelvis[:, 1:2], pelvis, 2 * pelvis[:, -1:] - pelvis[:, -2:-1]), dim=1)
    return (padded[:, 2:] - padded[:, :-2]) * (NTU2P_FPS / 2.0)


def to_local(vectors, frames):
    """世界向量 [...,3] -> 朝向系坐标（R^T v），frames [...,3,3] 与 vectors 前缀对齐。"""
    # torch 1.7 的 einsum 不广播省略维，用 matmul：(v^T R)_j = Σ_i v_i R_ij。
    return (vectors.unsqueeze(-2) @ frames).squeeze(-2)


def to_world(vectors, frames):
    return (frames @ vectors.unsqueeze(-1)).squeeze(-1)


def leg_features(xyz, up, frames=None):
    """[B,T,P,55,3] -> 特征 [B,T,P,21]（膝/踝/脚掌相对骨盆 + 骨盆速度，均在本人朝向系）与朝向 R。"""
    if frames is None:
        frames = heading_frames(xyz, up)
    rel = xyz[..., list(GEN_JOINTS), :] - xyz[..., PELVIS : PELVIS + 1, :]
    rel = to_local(rel, frames.unsqueeze(-3))
    vel = to_local(pelvis_velocity(xyz), frames)
    return torch.cat((rel.flatten(-2), vel), dim=-1), frames


def pelvis_height(xyz, obs_xyz, up):
    """[B,T,P,1]：骨盆沿 up 相对本人观测期最低脚关节的高度（推理时只用观测估计地面）。"""
    ground = observed_ground(obs_xyz, up)
    return ((xyz[..., PELVIS, :] * up[:, None, None, :]).sum(-1) - ground[:, None]).unsqueeze(-1)


def leg_directions(xyz, frames):
    """[B,T,P,55,3] -> 6 段骨在本人朝向系下的单位方向 [B,T,P,6,3]（顺序同 bone_lengths）。"""
    bones = [xyz[..., child, :] - xyz[..., parent, :] for chain in CHAINS for parent, child in zip(chain[:-1], chain[1:])]
    return _unit(to_local(torch.stack(bones, dim=-2), frames.unsqueeze(-3)))


def bone_lengths(xyz_frame):
    """[B,P,55,3] -> [B,P,6]，顺序 (L 大腿, L 小腿, L 脚, R 大腿, R 小腿, R 脚)。"""
    out = []
    for chain in CHAINS:
        for parent, child in zip(chain[:-1], chain[1:]):
            out.append((xyz_frame[..., child, :] - xyz_frame[..., parent, :]).norm(dim=-1))
    return torch.stack(out, dim=-1)


def fk_legs(hips, directions, lengths, frames):
    """hips [B,T,P,2,3]、directions [B,T,P,6,3]（朝向系单位向量）、lengths [B,P,6] -> 关节 [B,T,P,6,3]（GEN_JOINTS 顺序）。"""
    world = to_world(_unit(directions), frames.unsqueeze(-3))
    joints = []
    for side in range(2):
        point = hips[..., side, :]
        for bone in range(3):
            index = side * 3 + bone
            point = point + lengths[:, None, :, index, None] * world[..., index, :]
            joints.append(point)
    return torch.stack(joints, dim=-2)


def dct_basis(length, order, device=None, dtype=torch.float32):
    t = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    k = torch.arange(order, device=device, dtype=dtype).unsqueeze(0)
    basis = torch.cos(math.pi * (t + 0.5) * k / length) * math.sqrt(2.0 / length)
    basis[:, 0] = basis[:, 0] / math.sqrt(2.0)
    return basis


class PeriodicAutoencoder(nn.Module):
    """DeepPhase 式 PAE：卷积得到 K 条潜曲线，FFT 给频率/振幅/偏置，线性层 + atan2 给相位，正弦重建后解码。

    曲线参数化为 a·sin(2π(f·T + ψ)) + b（T 以窗口中心为 0），窗口中心前移一帧时 ψ 增加约 f/FPS。
    """

    def __init__(self, in_dim=FEAT_DIM, channels=4, window=41, hidden=64):
        super().__init__()
        self.channels, self.window = int(channels), int(window)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 9, padding=4), nn.ELU(), nn.Conv1d(hidden, hidden, 9, padding=4), nn.ELU(),
            nn.Conv1d(hidden, channels, 9, padding=4),
        )
        self.phase_weight = nn.Parameter(torch.randn(channels, window, 2) / math.sqrt(window))
        self.phase_bias = nn.Parameter(torch.zeros(channels, 2))
        self.decoder = nn.Sequential(
            nn.Conv1d(channels, hidden, 9, padding=4), nn.ELU(), nn.Conv1d(hidden, hidden, 9, padding=4), nn.ELU(),
            nn.Conv1d(hidden, in_dim, 9, padding=4),
        )
        self.register_buffer("time", (torch.arange(window, dtype=torch.float32) - window // 2) / NTU2P_FPS)
        self.register_buffer("freqs", torch.arange(window // 2 + 1, dtype=torch.float32) * NTU2P_FPS / window)

    def params(self, x):
        """x [N,in,W] -> OrderedDict(f, a, b, psi) 各 [N,K]。"""
        curves = self.encoder(x)
        spectrum = torch.fft.rfft(curves, dim=-1)
        power = spectrum.real.pow(2) + spectrum.imag.pow(2)
        ac = power[..., 1:]
        freq = (self.freqs[1:] * ac).sum(-1) / ac.sum(-1).clamp_min(1e-8)
        amp = 2.0 * ac.sum(-1).clamp_min(1e-12).sqrt() / self.window
        offset = spectrum.real[..., 0] / self.window
        sxy = torch.einsum("nkw,kwc->nkc", curves, self.phase_weight) + self.phase_bias
        psi = torch.atan2(sxy[..., 1], sxy[..., 0]) / (2 * math.pi)
        return OrderedDict([("f", freq), ("a", amp), ("b", offset), ("psi", psi)])

    def reconstruct(self, p):
        curve = p["a"][..., None] * torch.sin(2 * math.pi * (p["f"][..., None] * self.time + p["psi"][..., None])) + p["b"][..., None]
        return self.decoder(curve)

    def forward(self, x):
        p = self.params(x)
        return self.reconstruct(p), p


def phase_vector(p):
    """逐帧相位流形坐标：[..., K] 参数 -> [..., 4K] = (a sin2πψ, a cos2πψ, f, b)。"""
    angle = 2 * math.pi * p["psi"]
    return torch.cat((p["a"] * torch.sin(angle), p["a"] * torch.cos(angle), p["f"], p["b"]), dim=-1)


class GaitDecoder(nn.Module):
    """G：逐帧相位坐标 + 骨盆速度 + 腿长 + 动作 -> 6 段骨方向（朝向系）与 4 个着地 logit；时间卷积看前后 4 帧。"""

    def __init__(self, channels=4, hidden=256, action_dim=16, ground_aware=False, obs_pose=False):
        super().__init__()
        self.ground_aware, self.obs_pose = bool(ground_aware), bool(obs_pose)
        self.action_embed = nn.Embedding(NUM_ACTIONS, action_dim)
        in_dim = 4 * channels + 3 + NUM_BONES + action_dim + int(self.ground_aware) + NUM_BONES * 3 * int(self.obs_pose)
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 5, padding=2), nn.ELU(), nn.Conv1d(hidden, hidden, 5, padding=2), nn.ELU(),
            nn.Conv1d(hidden, NUM_BONES * 3 + len(CONTACT_JOINTS), 1),
        )
        # 方向偏置初始化为站立：大腿/小腿朝下，脚朝前下，使初期 FK 不出现折叠腿。
        rest = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.7, -0.7]] * 2).flatten()
        with torch.no_grad():
            self.net[-1].bias[: NUM_BONES * 3].copy_(rest)

    def forward(self, phase, velocity, lengths, action, height=None, obs_dirs=None):
        """phase [B,T,P,4K]、velocity [B,T,P,3]（朝向系）、lengths [B,P,6]、action [B]、height [B,T,P,1]（骨盆离观测地面，
        ground_aware 时必给）-> (dirs [B,T,P,6,3], logits [B,T,P,4])。"""
        batch, frames, persons = phase.shape[:3]
        emb = self.action_embed(action)[:, None, None, :].expand(batch, frames, persons, -1)
        length = lengths[:, None].expand(batch, frames, persons, NUM_BONES)
        parts = (phase, velocity, length, emb) + ((height - PELVIS_HEIGHT_REF,) if self.ground_aware else ())
        if self.obs_pose:
            parts += (obs_dirs.flatten(-2)[:, None].expand(batch, frames, persons, -1),)
        x = torch.cat(parts, dim=-1)
        x = x.permute(0, 2, 3, 1).reshape(batch * persons, -1, frames)
        y = self.net(x).reshape(batch, persons, -1, frames).permute(0, 3, 1, 2)
        return y[..., : NUM_BONES * 3].reshape(batch, frames, persons, NUM_BONES, 3), y[..., NUM_BONES * 3 :]


class PhaseForecaster(nn.Module):
    """F：观测腿部特征 + 未来骨盆速度 + 动作 -> 末帧相位 ψ0 与未来 f/a/b 轨迹（低阶 DCT），ψ 由频率积分。"""

    def __init__(self, channels=4, obs_len=10, pred_len=50, order=8, hidden=512, action_dim=16):
        super().__init__()
        self.channels, self.obs_len, self.pred_len, self.order = int(channels), int(obs_len), int(pred_len), int(order)
        self.action_embed = nn.Embedding(NUM_ACTIONS, action_dim)
        in_dim = obs_len * FEAT_DIM + pred_len * 3 + action_dim
        out_dim = channels * (2 + 3 * order)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, out_dim))
        self.register_buffer("basis", dct_basis(pred_len, order))

    def forward(self, obs_feat, future_vel, action):
        """obs_feat [B,To,P,21]（已标准化）、future_vel [B,Tp,P,3] -> 未来 [B,Tp,P,K] 的 f/a/b/psi 与末帧 psi0 [B,P,K]。"""
        batch, persons = obs_feat.shape[0], obs_feat.shape[2]
        emb = self.action_embed(action)[:, None, :].expand(batch, persons, -1)
        x = torch.cat((obs_feat.permute(0, 2, 1, 3).flatten(2), future_vel.permute(0, 2, 1, 3).flatten(2), emb), dim=-1)
        out = self.net(x).view(batch, persons, self.channels, 2 + 3 * self.order)
        psi0 = torch.atan2(out[..., 1], out[..., 0]) / (2 * math.pi)
        coef = out[..., 2:].view(batch, persons, self.channels, 3, self.order)
        traj = torch.einsum("tk,bpcmk->btpcm", self.basis, coef)
        freq = F.softplus(traj[..., 0] + 1.0)
        amp = F.softplus(traj[..., 1])
        offset = traj[..., 2]
        psi = psi0[:, None] + torch.cumsum(freq, dim=1) / NTU2P_FPS
        return OrderedDict([("f", freq), ("a", amp), ("b", offset), ("psi", psi)]), psi0


def extend_back(params, psi0, obs_len):
    """观测帧的相位坐标：从末帧按首个预测频率反推、振幅/偏置取首帧值，供 G 的时间卷积看到连续上下文。"""
    steps = torch.arange(obs_len - 1, -1, -1, device=psi0.device, dtype=psi0.dtype).view(1, obs_len, 1, 1)
    first = OrderedDict((k, v[:, :1]) for k, v in params.items())
    back = OrderedDict()
    back["f"] = first["f"].expand(-1, obs_len, -1, -1)
    back["a"] = first["a"].expand(-1, obs_len, -1, -1)
    back["b"] = first["b"].expand(-1, obs_len, -1, -1)
    back["psi"] = psi0[:, None] - steps * first["f"] / NTU2P_FPS
    return OrderedDict((k, torch.cat((back[k], params[k]), dim=1)) for k in params)


class PhaseGaitHead(nn.Module):
    """推理整体：PAE 只在训练时提供 GT 相位；这里持有 F、G 与特征标准化统计。"""

    def __init__(self, config):
        super().__init__()
        self.config = OrderedDict(config)
        k = int(config["channels"])
        self.forecaster = PhaseForecaster(channels=k, obs_len=config["obs_len"], pred_len=config["pred_len"], order=config["order"])
        self.decoder = GaitDecoder(channels=k, ground_aware=bool(config.get("ground_aware", False)), obs_pose=bool(config.get("obs_pose_input", False)))
        self.anchor_tau = float(config.get("anchor_tau", 0.0))
        self.register_buffer("feat_mean", torch.zeros(FEAT_DIM))
        self.register_buffer("feat_std", torch.ones(FEAT_DIM))
        self.register_buffer("phase_mean", torch.zeros(4 * k))
        self.register_buffer("phase_std", torch.ones(4 * k))

    def norm_phase(self, vector):
        return (vector - self.phase_mean) / self.phase_std

    def legs(self, phase_params, velocity, hips, lengths, frames, action, height=None, obs_dirs=None):
        """相位参数 [B,T,P,K] -> (腿关节 [B,T,P,6,3], 着地 logit [B,T,P,4])；obs_dirs 为观测末帧骨方向 [B,P,6,3]。"""
        dirs, logits = self.decoder(self.norm_phase(phase_vector(phase_params)), velocity, lengths, action, height, obs_dirs)
        if self.anchor_tau > 0:
            # 观测锚定：与观测末帧的方向差按 exp(-Δt/τ) 衰减，首帧严格连续，之后过渡到相位驱动的步态。
            t0 = int(self.config["obs_len"]) - 1
            dirs = _unit(dirs)
            residual = obs_dirs - dirs[:, t0]
            steps = (torch.arange(dirs.shape[1], device=dirs.device, dtype=dirs.dtype) - t0).clamp_min(0)
            weight = torch.exp(-steps / self.anchor_tau).view(1, -1, 1, 1, 1)
            dirs = dirs + residual[:, None] * weight
        return fk_legs(hips, dirs, lengths, frames), logits

    def forward(self, obs_xyz, pred_xyz, action, up):
        """obs [B,To,2,55,3]、主体预测 pred [B,Tp,2,55,3] -> 替换膝/踝/脚掌后的预测。"""
        full = torch.cat((obs_xyz, pred_xyz), dim=1).float()
        feat, frames = leg_features(full, up.float())
        obs_len = int(obs_xyz.shape[1])
        obs_feat = (feat[:, :obs_len] - self.feat_mean) / self.feat_std
        velocity = feat[..., -3:]
        params, psi0 = self.forecaster(obs_feat, velocity[:, obs_len:], action)
        full_params = extend_back(params, psi0, obs_len)
        lengths = bone_lengths(obs_xyz[:, -1].float())
        height = pelvis_height(full, obs_xyz.float(), up.float())
        obs_dirs = leg_directions(full[:, obs_len - 1 : obs_len], frames[:, obs_len - 1 : obs_len])[:, 0]
        joints, _ = self.legs(full_params, velocity, full[..., [L_HIP, R_HIP], :], lengths, frames, action, height, obs_dirs)
        out = pred_xyz.clone()
        out[..., list(GEN_JOINTS), :] = joints[:, obs_len:].to(pred_xyz.dtype)
        return out


def load_gait_head(path, device):
    state = torch.load(path, map_location="cpu")
    head = PhaseGaitHead(state["config"])
    head.load_state_dict(state["head"])
    return head.to(device).eval()


__all__ = [
    "CHAINS", "CONTACT_JOINTS", "FEAT_DIM", "GEN_JOINTS", "GaitDecoder", "PeriodicAutoencoder", "PhaseForecaster", "PhaseGaitHead",
    "bone_lengths", "dct_basis", "extend_back", "leg_directions", "pelvis_height", "fk_legs", "heading_frames", "leg_features", "load_gait_head", "pelvis_velocity", "phase_vector",
]
