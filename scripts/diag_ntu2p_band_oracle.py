"""第 0a 阶段：流状态局部 DCT 阶数 K 的 oracle 代价（只读 train）。

GT 未来经 NTU2PJiTFlow 的同一编解码（ramp-DCT 最小二乘 → IDCT×ramp → 手指模板 → 骨架投影）后与 GT 比较，
另报 GT 本身在评价口径三频带上的局部姿态能量，用来核对"k≥11 多为拟合抖动"的量级。
设计与判定规则：docs/ai/context/20261103-215324-ntu2p-diffusion-full-prediction-design-and-plan.md 3.1 节。
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, eval_windows, iter_eval_batches
from model.forecasting_ntu2p_jitflow import NTU2PJiTFlow
from utils.ntu_smplx_2p_xyz import compute_ntu_xyz_metrics, dct_band_energies

# 设计 3.1：Δoracle xyz_mse 超过 FD2 H-val xyz_mse 的 2% 时改用 K=14。
FD2_HVAL_XYZ_MSE = 0.019382804921261373
FALLBACK_FRACTION = 0.02


def oracle(cache, local_k, batch_size, device):
    model = NTU2PJiTFlow(denoiser="stg", local_k=local_k).to(device).eval()
    totals, count = OrderedDict(), 0
    with torch.no_grad():
        for batch in iter_eval_batches(eval_windows(cache), batch_size):
            obs, target = batch["obs_xyz"].to(device), batch["target_xyz"].to(device)
            prep = model.prepare(obs, batch["action"].to(device))
            pred, _, _ = model.decode_xyz(prep, *model.encode_target(prep, target))
            size = int(obs.shape[0])
            for key, value in compute_ntu_xyz_metrics(pred, target, obs).items():
                if key in ("xyz_mse", "xyz_mae", "mpjpe"):
                    totals[key] = totals.get(key, 0.0) + float(value) * size
            for band, value in dct_band_energies(target).items():
                totals["gt_dct_" + band] = totals.get("gt_dct_" + band, 0.0) + float(value.sum())
            count += size
    return OrderedDict((key, value / count) for key, value in totals.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", default="results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json")
    parser.add_argument("--cache_dir", default="results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval")
    parser.add_argument("--ks", type=int, nargs="+", default=[11, 14, 20])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/forecasting/ntu120_label/ntu2p_bandflow/oracle_train.json")
    opts = parser.parse_args()
    device = torch.device(opts.device)
    cache = NTU2PXYZSeqCache.load(opts.cache_dir, "train", manifest_path=opts.manifest_path, device=device)
    rows = OrderedDict(("K{}".format(k), oracle(cache, k, opts.batch_size, device)) for k in opts.ks)
    full = rows["K{}".format(max(opts.ks))]
    limit = FALLBACK_FRACTION * FD2_HVAL_XYZ_MSE
    for row in rows.values():
        row["delta_xyz_mse_vs_full"] = row["xyz_mse"] - full["xyz_mse"]
    chosen = 11 if rows["K11"]["delta_xyz_mse_vs_full"] <= limit else 14
    result = OrderedDict([("split", "train"), ("windows", len(cache)), ("fallback_limit", limit), ("chosen_local_k", chosen), ("rows", rows)])
    Path(opts.output).parent.mkdir(parents=True, exist_ok=True)
    Path(opts.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
