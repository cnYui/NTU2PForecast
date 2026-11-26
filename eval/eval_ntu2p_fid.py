"""NTU2P 各方法的 FID（ReGenNet 同款 STGCN 特征 + Fréchet 距离），与 mpjpe、步态指标同表报告。

两个阶段：
- sensitivity：只用真实数据及其人为扰动（S0 噪声底、S1 腿随骨盆滑行、S2 抖动、S3 copy-last）与训练集参照，
  回答 FID 能看见什么；须在看任何方法的 FID 之前运行并记录。
- methods：读各方法已导出的 test 预测数组，算 FID、配对 bootstrap 置信区间、识别准确率与 mpjpe/步态指标。
设计与预登记的解读规则见 docs/ai/context/20261126-105120-ntu2p-fid-evaluation-design-and-plan.md。
"""

import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from scipy import linalg

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, eval_windows
from eval.eval_ntu2p_v2 import naturalness_block
from model.ntu2p_stgcn import load_recognizer
from train.train_ntu2p_fid_recognizer import ORIGINAL_CACHE, ORIGINAL_MANIFEST, classification_metrics, predict, sha256
from utils.ntu_smplx_2p_xyz import compute_ntu_xyz_metrics

LEG_JOINTS = (4, 5, 7, 8, 10, 11)  # 双膝、双踝、双脚，即 PGait 生成的 6 个关节
PELVIS = 0
JITTER_SIGMA = 0.01
TABLE_KEYS = ("mpjpe", "xyz_mse", "fid", "top1_acc", "slide_frame_ratio", "step_count", "jerk_body")


def gaussian_stats(features):
    features = np.asarray(features, dtype=np.float64)
    return features.mean(axis=0), np.cov(features, rowvar=False)


def sym_sqrt(sigma):
    values, vectors = np.linalg.eigh((sigma + sigma.T) / 2.0)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def frechet(mu1, sigma1, mu2, sigma2, root1=None):
    """d² = |mu1-mu2|² + Tr(S1 + S2 - 2 (S1 S2)^½)；Tr (S1 S2)^½ = Σ √eig(S1^½ S2 S1^½)，对称特征分解比
    scipy.linalg.sqrtm 稳定且快一个量级，bootstrap 时 S1^½ 只算一次。"""
    root1 = sym_sqrt(sigma1) if root1 is None else root1
    inner = root1 @ sigma2 @ root1
    trace_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((inner + inner.T) / 2.0), 0.0, None)).sum()
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * trace_sqrt)


def frechet_scipy(mu1, sigma1, mu2, sigma2):
    """ReGenNet eval/a2m/stgcn/fid.py 的原实现，只用于自检。"""
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fid(features_a, features_b):
    return frechet(*gaussian_stats(features_b), *gaussian_stats(features_a))


def sliding_legs(obs_xyz, target_xyz):
    """S1：6 个腿关节保持观测末帧相对骨盆的位置、随骨盆平移，即"腿不迈步、整体滑行"。"""
    offset = obs_xyz[:, -1:, :, list(LEG_JOINTS)] - obs_xyz[:, -1:, :, PELVIS : PELVIS + 1]
    out = target_xyz.clone()
    out[:, :, :, list(LEG_JOINTS)] = target_xyz[:, :, :, PELVIS : PELVIS + 1] + offset
    return out


def copy_last(obs_xyz, pred_len):
    return obs_xyz[:, -1:].expand(-1, int(pred_len), -1, -1, -1).contiguous()


def _sample_ids(meta):
    return [str(m["sample_id"]) if isinstance(m, dict) else str(m) for m in meta]


def load_entry(spec):
    """'名称@种子=路径#键'；本项目数组取 methods[键]（默认 model），ReGenNet 方法 1 数组取 键（默认 pred_xyz）。"""
    name, rest = spec.split("=", 1)
    path, _, key = rest.partition("#")
    data = torch.load(path, map_location="cpu")
    future = data["methods"][key or "model"] if "methods" in data else data[key or "pred_xyz"]
    return OrderedDict([("name", name), ("path", path), ("key", key), ("obs_xyz", data["obs_xyz"].float()),
                        ("target_xyz", data["target_xyz"].float()), ("future", future.float()),
                        ("actions", data["actions"].long().reshape(-1)), ("ids", _sample_ids(data["meta"]))])


def align(entry, reference):
    """按 sample_id 对齐到参照顺序，并核对观测与 GT 一致，防止不同导出的窗口错位。"""
    if entry["ids"] != reference["ids"]:
        position = {sid: i for i, sid in enumerate(entry["ids"])}
        if sorted(position) != sorted(reference["ids"]):
            raise ValueError("{} 的样本集合与参照不同".format(entry["name"]))
        order = torch.tensor([position[sid] for sid in reference["ids"]])
        for key in ("obs_xyz", "target_xyz", "future", "actions"):
            entry[key] = entry[key][order]
        entry["ids"] = list(reference["ids"])
    for key in ("obs_xyz", "target_xyz"):
        gap = float((entry[key] - reference[key]).abs().max())
        if gap > 1e-3:
            raise ValueError("{} 的 {} 与参照最大相差 {:.4g} m".format(entry["name"], key, gap))
    if not torch.equal(entry["actions"], reference["actions"]):
        raise ValueError("{} 的动作标签与参照不同".format(entry["name"]))
    return entry


def recognizer_info(path):
    return OrderedDict([("path", path), ("sha256", sha256(path))])


def sensitivity(args, recognizer, device, reference):
    obs, target = reference["obs_xyz"], reference["target_xyz"]
    generator = torch.Generator().manual_seed(int(args.seed))
    variants = OrderedDict(
        [
            ("S1_sliding_legs", sliding_legs(obs, target)),
            ("S2_jitter_1cm", target + JITTER_SIGMA * torch.randn(target.shape, generator=generator)),
            ("S3_copy_last", copy_last(obs, target.shape[1])),
        ]
    )
    _, gt_feat = predict(recognizer, obs, target, device)
    feats = OrderedDict((name, predict(recognizer, obs, value, device)[1]) for name, value in variants.items())
    gt_np = gt_feat.numpy()
    rng = np.random.default_rng(int(args.seed))
    n = gt_np.shape[0]
    half = OrderedDict([("S0_gt_vs_gt", [])] + [(name, []) for name in variants])
    for _ in range(int(args.num_splits)):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2 :]
        mu_b, sigma_b = gaussian_stats(gt_np[b])
        root_b = sym_sqrt(sigma_b)
        half["S0_gt_vs_gt"].append(frechet(mu_b, sigma_b, *gaussian_stats(gt_np[a]), root1=root_b))
        for name, value in feats.items():
            half[name].append(frechet(mu_b, sigma_b, *gaussian_stats(value.numpy()[a]), root1=root_b))
    floor = float(np.percentile(half["S0_gt_vs_gt"], 95))
    result = OrderedDict()
    for name, values in half.items():
        values = np.asarray(values)
        result[name] = OrderedDict([("half_median", float(np.median(values))), ("half_p05", float(np.percentile(values, 5))),
                                    ("half_p95", float(np.percentile(values, 95)))])
        if name != "S0_gt_vs_gt":
            result[name]["full_fid"] = fid(feats[name].numpy(), gt_np)
            result[name]["visible"] = bool(result[name]["half_median"] > floor)

    train_windows = eval_windows(NTU2PXYZSeqCache.load(args.train_cache_dir, "train", args.train_manifest_path, device="cpu"))
    _, train_feat = predict(recognizer, train_windows["obs_xyz"], train_windows["target_xyz"], device)
    checks = OrderedDict()
    checks["fid_eig_vs_scipy_train_vs_test"] = [fid(train_feat.numpy(), gt_np),
                                                frechet_scipy(*gaussian_stats(gt_np), *gaussian_stats(train_feat.numpy()))]
    checks["fid_gt_vs_gt"] = fid(gt_np, gt_np)
    return OrderedDict(
        [
            ("stage", "sensitivity"),
            ("num_samples", int(n)),
            ("num_splits", int(args.num_splits)),
            ("noise_floor_p95", floor),
            ("variants", result),
            ("train_gt_center_windows_vs_test_gt_fid", fid(train_feat.numpy(), gt_np)),
            ("num_train_windows", int(train_feat.shape[0])),
            ("checks", checks),
        ]
    )


def method_table(args, recognizer, device, entries):
    reference = entries[0]
    obs, target, actions = reference["obs_xyz"], reference["target_xyz"], reference["actions"]
    _, gt_feat = predict(recognizer, obs, target, device)
    gt_np = gt_feat.numpy()
    rows, feats = OrderedDict(), OrderedDict()
    for entry in entries:
        logits, feat = predict(recognizer, obs, entry["future"], device)
        feats[entry["name"]] = feat.numpy()
        row = OrderedDict([("path", entry["path"]), ("key", entry["key"])])
        row.update((k, v) for k, v in compute_ntu_xyz_metrics(entry["future"], target, obs).items() if k in ("mpjpe", "xyz_mse", "xyz_mae"))
        row["fid"] = fid(feats[entry["name"]], gt_np)
        accuracy = classification_metrics(logits, actions)
        row["top1_acc"], row["balanced_acc"] = accuracy["top1_acc"], accuracy["balanced_acc"]
        rows[entry["name"]] = row

    variants = OrderedDict([("GT", target), ("copy_last", copy_last(obs, target.shape[1]))])
    variants.update((entry["name"], entry["future"]) for entry in entries)
    natural = naturalness_block(obs, target, variants)
    for name in rows:
        rows[name].update((key, float(natural[name][key])) for key in ("slide_frame_ratio", "step_count", "jerk_body"))
    gt_natural = OrderedDict((key, float(natural["GT"][key])) for key in ("slide_frame_ratio", "step_count", "jerk_body"))

    rng = np.random.default_rng(int(args.seed))
    n = gt_np.shape[0]
    boot = OrderedDict((name, []) for name in feats)
    for _ in range(int(args.num_bootstrap)):
        index = rng.integers(0, n, n)
        mu_gt, sigma_gt = gaussian_stats(gt_np[index])
        root_gt = sym_sqrt(sigma_gt)
        for name, value in feats.items():
            boot[name].append(frechet(mu_gt, sigma_gt, *gaussian_stats(value[index]), root1=root_gt))
    for name, values in boot.items():
        rows[name]["fid_ci95"] = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    pairs = OrderedDict()
    for spec in args.pair:
        a, b = spec.split(":")
        diff = np.asarray(boot[a]) - np.asarray(boot[b])
        pairs["{}-{}".format(a, b)] = OrderedDict([("delta_fid", rows[a]["fid"] - rows[b]["fid"]),
                                                   ("ci95", [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))])])
    return OrderedDict([("rows", rows), ("pairs", pairs), ("gt_naturalness", gt_natural), ("num_samples", int(n)),
                        ("num_bootstrap", int(args.num_bootstrap))])


def aggregate(rows):
    """'方法@种子' -> 方法：各列 3 种子均值 ± 标准差（总体标准差）。"""
    groups = OrderedDict()
    for name, row in rows.items():
        groups.setdefault(name.split("@")[0], []).append(row)
    out = OrderedDict()
    for method, members in groups.items():
        out[method] = OrderedDict([("num_seeds", len(members))])
        for key in TABLE_KEYS:
            values = np.asarray([m[key] for m in members], dtype=np.float64)
            out[method][key] = [float(values.mean()), float(values.std())]
    return out


def markdown(result):
    lines = ["| 方法 | 种子数 | " + " | ".join(TABLE_KEYS) + " |", "|---|---:|" + "---:|" * len(TABLE_KEYS)]
    for method, row in result["aggregate"].items():
        cells = []
        for key in TABLE_KEYS:
            mean, std = row[key]
            cells.append("{:.4g}".format(mean) if row["num_seeds"] == 1 else "{:.4g} ± {:.2g}".format(mean, std))
        lines.append("| {} | {} | {} |".format(method, row["num_seeds"], " | ".join(cells)))
    gt = result["gt_naturalness"]
    lines.append("| GT | — | 0 | 0 | 0 | — | {:.4f} | {:.4f} | {:.4f} |".format(gt["slide_frame_ratio"], gt["step_count"], gt["jerk_body"]))
    lines += ["", "| 配对 | ΔFID | 95% CI |", "|---|---:|---|"]
    for pair, value in result["pairs"].items():
        lines.append("| {} | {:+.4g} | [{:+.4g}, {:+.4g}] |".format(pair, value["delta_fid"], *value["ci95"]))
    return "\n".join(lines) + "\n"


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，不回退 CPU；冒烟测试请显式 --device cpu")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    recognizer = load_recognizer(args.recognizer, device)
    if args.stage == "sensitivity":
        result = sensitivity(args, recognizer, device, eval_windows(NTU2PXYZSeqCache.load(ORIGINAL_CACHE, "test", ORIGINAL_MANIFEST)))
    else:
        entries = [load_entry(spec) for spec in args.entry]
        entries = [entries[0]] + [align(entry, entries[0]) for entry in entries[1:]]
        # copy-last 不需要导出：由参照的观测末帧直接构造。
        copy_entry = OrderedDict(entries[0])
        copy_entry.update(name="copy-last", path="", key="obs_last",
                          future=copy_last(copy_entry["obs_xyz"], copy_entry["target_xyz"].shape[1]))
        result = method_table(args, recognizer, device, [copy_entry] + entries)
        result["aggregate"] = aggregate(result["rows"])
    result["recognizer"] = recognizer_info(args.recognizer)
    result["created_at"] = datetime.utcnow().isoformat() + "Z"
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if args.stage == "methods":
        with open(os.path.splitext(args.output)[0] + ".md", "w") as f:
            f.write(markdown(result))
        print(markdown(result))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sensitivity", "methods"), required=True)
    parser.add_argument("--recognizer", required=True)
    parser.add_argument("--entry", action="append", default=[], help="'名称@种子=数组路径#键'，第一个作为 GT/观测参照")
    parser.add_argument("--pair", action="append", default=[], help="'名称@种子:名称@种子'，报 ΔFID 的配对 bootstrap CI")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train_cache_dir", default=ORIGINAL_CACHE)
    parser.add_argument("--train_manifest_path", default=ORIGINAL_MANIFEST)
    parser.add_argument("--num_splits", type=int, default=100)
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
