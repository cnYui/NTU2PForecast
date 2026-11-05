"""落脚规划层：把模型输出里随骨盆漂移的脚改成"着地锁定 + 离散迈步"，再用两骨 IK 重解膝。

设计与参数来源：docs/ai/context/20261104-170558-ntu2p-foot-planner-design-and-plan.md。
确定性读出把不同的步相位平均掉，脚的运动塌缩成骨盆平移（滑行）；这里对每只脚选定落脚时刻与落点：
着地期间踝与脚掌固定，被身体带离锁定点超过 trigger 时按步长自适应的帧数划弧落到预测位置前方 lead 处，同一时刻
只有一只脚在空中。只改双腿的膝、踝、脚掌；髋、骨盆与上半身逐位不变。参数全部来自 train GT 的步态统计。
"""

import math
from collections import OrderedDict

import torch

from utils.ntu2p_canonical import estimate_up

L_HIP, R_HIP = 1, 2
L_KNEE, R_KNEE = 4, 5
L_ANKLE, R_ANKLE = 7, 8
L_FOOT, R_FOOT = 10, 11
LEGS = ((L_HIP, L_KNEE, L_ANKLE, L_FOOT), (R_HIP, R_KNEE, R_ANKLE, R_FOOT))

PLANTED, SWING, FREE = 0, 1, 2
# train GT 步行人中位数（scripts/diag_ntu2p_gt_gait_stats.py）：离地时踝在骨盆后 0.158 m、落地时在前 0.102 m，
# 摆动抬高 0.070 m，去抖后内部着地段 5 帧；free_height 高于摆动抬高 p75（0.136 m）。
# 摆动时长按步长自适应：踝峰速/root 均速 4.277，smoothstep 峰速为均速 1.5 倍 → 摆动脚均速 ≈ 2.85 × 身体速度；
# 踝速 > 1.5×root 的连续段 p50 7、p75 12 帧。接触阈值口径的 21 帧被抖动高估，会把后腿拉成弓步（修订文档）。
DEFAULT_PARAMS = OrderedDict(
    [("trigger", 0.16), ("lead", 0.10), ("lift", 0.07), ("min_stance", 5), ("free_height", 0.20), ("replant_height", 0.05),
     ("swing_speed_ratio", 2.85), ("swing_guess", 7), ("swing_min", 5), ("swing_max", 12), ("min_body_speed", 0.10)]
)


def _height(value, up):
    return (value * up).sum(-1)


def _flat(value, up):
    return value - _height(value, up).unsqueeze(-1) * up


def _unit(value, eps=1e-9):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)


def _plan(pred, obs, up, p):
    """逐帧状态机 -> 计划的踝/脚掌位置 [B,T,2,2,3] 与是否自由 [B,T,2,2]。"""
    batch, frames = int(pred.shape[0]), int(pred.shape[1])
    ankles, toes = [leg[2] for leg in LEGS], [leg[3] for leg in LEGS]
    up4 = up[:, None, None, :]  # [B,1,1,3] 对齐 [B,人,脚,3]
    pred_ankle, pred_toe = pred[:, :, :, ankles], pred[:, :, :, toes]  # [B,T,2,2,3]
    obs_ankle, obs_toe = obs[:, :, :, ankles], obs[:, :, :, toes]
    rest = _height(obs_ankle, up[:, None, None, None, :]).min(dim=1).values  # [B,2,2]
    toe_rest = _height(obs_toe, up[:, None, None, None, :]).min(dim=1).values

    def at_rest(point, height):
        return _flat(point, up4) + height.unsqueeze(-1) * up4

    last_ankle, last_toe = obs_ankle[:, -1], obs_toe[:, -1]
    planted0 = _height(last_ankle, up4) - rest < p["replant_height"]
    mode = torch.where(planted0, torch.full_like(rest, PLANTED), torch.full_like(rest, FREE)).long()
    lock, lock_toe = last_ankle.clone(), last_toe.clone()
    stance = torch.full_like(rest, float(p["min_stance"]))  # 观测末帧已着地多久未知，视为已满足最短着地
    start, start_toe, target, target_toe = lock.clone(), lock_toe.clone(), lock.clone(), lock_toe.clone()
    progress = torch.zeros_like(rest)
    swing_len = torch.full_like(rest, float(p["swing_guess"]))
    velocity_last = pred_ankle[:, -1] - pred_ankle[:, -2]
    pelvis = _flat(pred[:, :, :, 0], up[:, None, None, :])  # [B,T,2,3]
    body_speed = torch.cat((pelvis[:, 1:2] - pelvis[:, 0:1], pelvis[:, 1:] - pelvis[:, :-1]), dim=1).norm(dim=-1) * 20.0  # [B,T,2]

    def predicted_at(index):
        """index [B,2,2] 整数帧号 -> 该帧预测的踝位置与"脚掌 − 踝"；超出窗口按末帧速度外推。"""
        clipped = index.clamp(max=frames - 1)
        gather = clipped.view(batch, 1, 2, 2, 1).expand(batch, 1, 2, 2, 3)
        ankle = pred_ankle.gather(1, gather).squeeze(1)
        toe = pred_toe.gather(1, gather).squeeze(1)
        extra = (index - clipped).to(ankle.dtype).unsqueeze(-1)
        return ankle + extra * velocity_last, toe - ankle

    out_ankle, out_toe, out_free = [], [], []
    for t in range(frames):
        ankle_now, toe_offset_now = pred_ankle[:, t], pred_toe[:, t] - pred_ankle[:, t]
        lifted = _height(ankle_now, up4) - rest
        # 预测里脚抬得很高（踢、跨）：不锁，跟随预测；回落后在当前位置重新着地。
        to_free = (lifted > p["free_height"]) & (mode != SWING)
        mode = torch.where(to_free, torch.full_like(mode, FREE), mode)
        replant = (mode == FREE) & (lifted < p["replant_height"])
        lock = torch.where(replant.unsqueeze(-1), at_rest(ankle_now, rest), lock)
        lock_toe = torch.where(replant.unsqueeze(-1), at_rest(ankle_now, rest) + toe_offset_now, lock_toe)
        stance = torch.where(replant, torch.zeros_like(stance), stance)
        mode = torch.where(replant, torch.full_like(mode, PLANTED), mode)

        drift = _flat(ankle_now - lock, up4).norm(dim=-1)  # [B,2,2]
        other_planted = (mode == PLANTED).flip(-1)
        eligible = (mode == PLANTED) & (stance >= p["min_stance"]) & (drift > p["trigger"]) & other_planted
        both = eligible[..., 0] & eligible[..., 1]
        left_first = drift[..., 0] >= drift[..., 1]
        eligible = torch.stack((eligible[..., 0] & (~both | left_first), eligible[..., 1] & (~both | ~left_first)), dim=-1)
        if bool(eligible.any()):
            guess = torch.full_like(rest, float(p["swing_guess"]))
            # 身体速度取该人预测骨盆在接下来 swing_guess 帧内的平均水平速度。
            window = body_speed[:, t : t + int(p["swing_guess"])].mean(dim=1).clamp_min(p["min_body_speed"])  # [B,2]
            speed = window.unsqueeze(-1).expand_as(rest)

            def target_for(length):
                landing, landing_offset = predicted_at(t + length.long() - 1)
                motion = _flat(landing - ankle_now, up4)
                fallback = _flat(ankle_now - lock, up4)
                direction = _unit(torch.where(motion.norm(dim=-1, keepdim=True) > 1e-3, motion, fallback))
                return at_rest(landing, rest) + p["lead"] * direction, landing_offset

            first, _ = target_for(guess)
            distance = _flat(first - lock, up4).norm(dim=-1)
            length = (20.0 * distance / (p["swing_speed_ratio"] * speed)).round().clamp(p["swing_min"], p["swing_max"])
            new_target, landing_offset = target_for(length)
            swing_len = torch.where(eligible, length, swing_len)
            pick = eligible.unsqueeze(-1)
            start, start_toe = torch.where(pick, lock, start), torch.where(pick, lock_toe, start_toe)
            target, target_toe = torch.where(pick, new_target, target), torch.where(pick, new_target + landing_offset, target_toe)
            progress = torch.where(eligible, torch.zeros_like(progress), progress)
            mode = torch.where(eligible, torch.full_like(mode, SWING), mode)

        swinging = mode == SWING
        progress = torch.where(swinging, progress + 1.0, progress)
        u = (progress / swing_len).clamp(0.0, 1.0)
        ease = (u * u * (3.0 - 2.0 * u)).unsqueeze(-1)
        arc = (p["lift"] * torch.sin(math.pi * u)).unsqueeze(-1) * up4
        swing_ankle = start + ease * (target - start) + arc
        swing_toe = start_toe + ease * (target_toe - start_toe) + arc
        landed = swinging & (progress >= swing_len)
        lock = torch.where(landed.unsqueeze(-1), target, lock)
        lock_toe = torch.where(landed.unsqueeze(-1), target_toe, lock_toe)
        mode = torch.where(landed, torch.full_like(mode, PLANTED), mode)
        stance = torch.where(landed, torch.zeros_like(stance), torch.where(mode == PLANTED, stance + 1.0, stance))

        free = (mode == FREE).unsqueeze(-1)
        planted = (mode == PLANTED).unsqueeze(-1)
        ankle_out = torch.where(free, ankle_now, torch.where(planted, lock, swing_ankle))
        toe_out = torch.where(free, pred_toe[:, t], torch.where(planted, lock_toe, swing_toe))
        out_ankle.append(ankle_out)
        out_toe.append(toe_out)
        out_free.append(mode == FREE)
    return torch.stack(out_ankle, dim=1), torch.stack(out_toe, dim=1), torch.stack(out_free, dim=1)


def _two_bone_ik(hip, knee_hint, ankle_target, thigh, shin):
    """髋固定、骨长固定：返回 (膝, 踝)。膝取解圆上离原预测膝最近的点；够不到时踝沿髋→目标方向取最大可达。"""
    offset = ankle_target - hip
    distance = offset.norm(dim=-1, keepdim=True)
    axis = _unit(offset)
    reach = torch.min(torch.max(distance, (thigh - shin).abs() + 1e-4), thigh + shin - 1e-4)
    ankle = hip + reach * axis
    along = (thigh * thigh - shin * shin + reach * reach) / (2.0 * reach)
    radius = (thigh * thigh - along * along).clamp_min(0.0).sqrt()
    center = hip + along * axis
    hint = knee_hint - center
    hint = hint - (hint * axis).sum(-1, keepdim=True) * axis
    return center + radius * _unit(hint), ankle


def plan_feet(pred_xyz, obs_xyz, params=None):
    """pred_xyz [B,T,2,55,3]、obs_xyz [B,10,2,55,3] -> 规划后的预测（同 dtype）与逐脚自由标记 [B,T,2,2]。"""
    p = OrderedDict(DEFAULT_PARAMS)
    p.update(params or {})
    pred, obs = pred_xyz.double(), obs_xyz.double()
    up = estimate_up(obs)
    ankle_plan, toe_plan, free = _plan(pred, obs, up, p)
    out = pred.clone()
    last = obs[:, -1:]  # 骨长取观测末帧，与骨架投影一致
    for side, (hip, knee, ankle, toe) in enumerate(LEGS):
        thigh = (last[..., knee, :] - last[..., hip, :]).norm(dim=-1, keepdim=True)
        shin = (last[..., ankle, :] - last[..., knee, :]).norm(dim=-1, keepdim=True)
        foot = (last[..., toe, :] - last[..., ankle, :]).norm(dim=-1, keepdim=True)
        new_knee, new_ankle = _two_bone_ik(pred[..., hip, :], pred[..., knee, :], ankle_plan[..., side, :], thigh, shin)
        new_toe = new_ankle + foot * _unit(toe_plan[..., side, :] - new_ankle)
        keep = free[..., side].unsqueeze(-1)
        out[..., knee, :] = torch.where(keep, pred[..., knee, :], new_knee)
        out[..., ankle, :] = torch.where(keep, pred[..., ankle, :], new_ankle)
        out[..., toe, :] = torch.where(keep, pred[..., toe, :], new_toe)
    return out.to(pred_xyz.dtype), free


__all__ = ["DEFAULT_PARAMS", "plan_feet"]
