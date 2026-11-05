"""诊断：流状态的 DCT 表示本身能否表达迈步（GT 经编解码后的步态指标），并与模型预测对照。

GT 未来经 NTU2PJiTFlow 的编解码（root/局部 ramp-DCT 最小二乘 → IDCT×ramp → 手指模板 → 骨架投影）后，
用 utils/ntu2p_naturalness 的步态指标与 GT 本身比较。若 oracle 已不迈步，任何在该表示上的模型都不可能迈步。
只读 val 的 GT 与已导出的预测数组，不训练。
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, eval_windows
from model.forecasting_ntu2p_jitflow import NTU2PJiTFlow
from utils.ntu2p_naturalness import aggregate_stats, compute_naturalness_stats

KEYS = ("mpjpe", "step_count", "slide_frame_ratio", "stance_fraction", "ankle_peak_to_root_speed", "foot_rel_motion_ratio",
        "lr_forward_vel_corr", "skate_gt_contact_speed_walk", "skate_self_ratio_walk", "leg_swing_amp", "gait_sep_corr_f01_10")


def summarize(pred, target, obs):
    stats = aggregate_stats(compute_naturalness_stats(pred, target, obs))
    row = OrderedDict((key, stats.get(key)) for key in KEYS if key != "mpjpe")
    row["mpjpe"] = float((pred.double() - target.double()).norm(dim=-1).mean())
    return row


def codec_roundtrip(obs, target, action, root_k, local_k, device):
    model = NTU2PJiTFlow(denoiser="stg", root_k=root_k, local_k=local_k).to(device).eval()
    outs = []
    with torch.no_grad():
        for begin in range(0, obs.shape[0], 64):
            o, t, a = obs[begin:begin + 64].to(device), target[begin:begin + 64].to(device), action[begin:begin + 64].to(device)
            prep = model.prepare(o, a)
            outs.append(model.decode_xyz(prep, *model.encode_target(prep, t))[0].cpu())
    return torch.cat(outs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", default="results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json")
    parser.add_argument("--cache_dir", default="results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval")
    parser.add_argument("--arrays", default="results/forecasting/ntu120_label/ntu2p_bandflow/review_s0/arrays.pt")
    parser.add_argument("--output", default="results/forecasting/ntu120_label/ntu2p_gait/oracle_val.json")
    parser.add_argument("--device", default="cuda:0")
    opts = parser.parse_args()
    device = torch.device(opts.device)
    windows = eval_windows(NTU2PXYZSeqCache.load(opts.cache_dir, "val", manifest_path=opts.manifest_path))
    obs, target, action = windows["obs_xyz"].double(), windows["target_xyz"].double(), windows["action"]
    arrays = torch.load(opts.arrays)
    if not torch.equal(arrays["target_xyz"].double(), target):
        raise ValueError("预测数组与 val 窗口不一致")
    rows = OrderedDict()
    rows["GT"] = summarize(target, target, obs)
    for root_k, local_k in ((8, 11), (8, 20), (8, 30), (20, 50)):
        rows["GT 经编解码 root{} local{}".format(root_k, local_k)] = summarize(codec_roundtrip(obs, target, action, root_k, local_k, device), target, obs)
    for name in arrays["methods"]:
        rows["预测 " + name] = summarize(arrays["methods"][name].double(), target, obs)
    Path(opts.output).parent.mkdir(parents=True, exist_ok=True)
    Path(opts.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    header = "| 行 | " + " | ".join(KEYS) + " |"
    print(header)
    for name, row in rows.items():
        print("| {} | {} |".format(name, " | ".join("—" if row[k] is None else "{:.3f}".format(row[k]) for k in KEYS)))


if __name__ == "__main__":
    main()
