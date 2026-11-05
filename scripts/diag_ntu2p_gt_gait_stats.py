"""train GT 的步态统计（只读 train）：摆动时长、步长、抬脚高度、步频与 root 速度，用于标定落脚规划层。

接触判定与评价口径相同（utils/ntu2p_naturalness.gt_contact_mask：平滑场景系下脚低于高度阈值且水平速度低于 0.2 m/s）；
一只脚算着地 = 踝或脚掌任一着地。只统计 GT 步行人（future 内 root 水平净位移 ≥ 0.5 m），另报全部人的着地占比。
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, eval_windows
from utils.ntu2p_naturalness import (
    FOOT_JOINTS,
    L_ANKLE,
    PELVIS,
    R_ANKLE,
    _smoothed_future_scene,
    estimate_scene_frame,
    gt_contact_mask,
    walk_mask,
)


# NTU 骨架拟合抖动会把接触切成 1–2 帧的碎片（未去抖时中位步长只有 7 cm），3 帧 = 0.15 s。
MIN_SEGMENT_FRAMES = 3
REAL_STEP_M = 0.15


def runs(mask):
    """布尔序列 -> [(start, stop)) 的 True 连续段。"""
    segments, start = [], None
    for t, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = t
        elif not value and start is not None:
            segments.append((start, t))
            start = None
    return segments


def debounce(mask, min_frames=MIN_SEGMENT_FRAMES):
    """先去掉短于 min_frames 的着地段，再填平短于 min_frames 的离地段（端点处的段不动，它们被窗口截断）。"""
    mask = mask.clone()
    last = len(mask)
    for start, stop in runs(mask):
        if stop - start < min_frames and start > 0 and stop < last:
            mask[start:stop] = False
    for start, stop in runs(~mask):
        if stop - start < min_frames and start > 0 and stop < last:
            mask[start:stop] = True
    return mask


def quantiles(values):
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    return OrderedDict((name, float(torch.quantile(tensor, q))) for name, q in (("p25", 0.25), ("p50", 0.5), ("p75", 0.75)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", default="results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json")
    parser.add_argument("--cache_dir", default="results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval")
    parser.add_argument("--output", default="results/forecasting/ntu120_label/ntu2p_gait/gt_gait_stats_train.json")
    opts = parser.parse_args()
    windows = eval_windows(NTU2PXYZSeqCache.load(opts.cache_dir, "train", manifest_path=opts.manifest_path))
    obs, target = windows["obs_xyz"].double(), windows["target_xyz"].double()
    frame = estimate_scene_frame(torch.cat((obs, target), dim=1))
    contact = gt_contact_mask(target, obs, frame)  # [B,T,2,4]：L_ANKLE, R_ANKLE, L_FOOT, R_FOOT
    planted = contact[..., :2] | contact[..., 2:]  # [B,T,2,2]
    walking = walk_mask(target, obs, frame)
    scene, velocity = _smoothed_future_scene(target, obs, frame)
    swing, step, lift, stance, cadence, speed = [], [], [], [], [], []
    # 只看"真正迈出去"的步（步长 ≥ REAL_STEP_M）：离地时踝在骨盆后方多远、落地时在前方多远、摆动帧数与平均速度。
    liftoff, landing, real_swing, swing_speed = [], [], [], []
    for b in range(obs.shape[0]):
        for p in range(2):
            if not bool(walking[b, p]):
                continue
            root_speed = velocity[b, :, p, PELVIS].norm(dim=-1).mean().item()
            disp = scene[b, -1, p, PELVIS, :2] - scene[b, 0, p, PELVIS, :2]
            fwd = disp / disp.norm().clamp_min(1e-6)
            steps_here = 0
            for f, ankle in enumerate((L_ANKLE, R_ANKLE)):
                segs = runs(debounce(planted[b, :, p, f]))
                for (s0, e0), (s1, e1) in zip(segs[:-1], segs[1:]):
                    swing.append(s1 - e0)
                    a0 = scene[b, s0:e0, p, ankle, :2].mean(0)
                    a1 = scene[b, s1:e1, p, ankle, :2].mean(0)
                    step.append(float((a1 - a0).norm()))
                    rest = scene[b, s0:e0, p, ankle, 2].mean()
                    lift.append(float(scene[b, e0:s1, p, ankle, 2].max() - rest))
                    steps_here += 1
                    if step[-1] >= REAL_STEP_M:
                        liftoff.append(float(((scene[b, e0 - 1, p, ankle, :2] - scene[b, e0 - 1, p, PELVIS, :2]) * fwd).sum()))
                        landing.append(float(((scene[b, s1, p, ankle, :2] - scene[b, s1, p, PELVIS, :2]) * fwd).sum()))
                        real_swing.append(s1 - e0)
                        swing_speed.append(step[-1] / ((s1 - e0) / 20.0))
                stance.extend(e - s for s, e in segs if s > 0 and e < planted.shape[1])
            cadence.append(steps_here / 2.5)
            speed.append(root_speed)
    result = OrderedDict(
        [
            ("split", "train"), ("windows", int(obs.shape[0])), ("walking_persons", int(walking.sum())),
            ("swing_frames", quantiles(swing)), ("stance_frames_interior", quantiles(stance)),
            ("step_length_m", quantiles(step)), ("ankle_lift_m", quantiles(lift)),
            ("steps_per_second_walking", quantiles(cadence)), ("root_speed_walking", quantiles(speed)),
            ("real_steps", len(liftoff)), ("liftoff_offset_fwd_m", quantiles(liftoff)), ("landing_offset_fwd_m", quantiles(landing)),
            ("real_swing_frames", quantiles(real_swing)), ("real_swing_speed_mps", quantiles(swing_speed)),
            ("planted_fraction_all_persons", float(planted.double().mean())),
            ("planted_fraction_walking", float(planted[walking[:, None, :, None].expand_as(planted)].double().mean())),
        ]
    )
    Path(opts.output).parent.mkdir(parents=True, exist_ok=True)
    Path(opts.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
