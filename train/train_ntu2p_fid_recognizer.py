"""训练 FID 用的双人 xyz STGCN 动作识别网络。

只用受试者留出划分的 train 训练、H-val 选 checkpoint；test 只用真实 GT 检查门槛，与任何预测方法无关。
优化器与超参数沿用 ReGenNet 识别网络（`recognition_training/ntu_smplx/opt.yaml`：AdamW、lr 1e-4、batch 64、
60 帧）。设计见 docs/ai/context/20261126-105120-ntu2p-fid-evaluation-design-and-plan.md。
"""

import argparse
import copy
import hashlib
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn.functional as F

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, NTU2PXYZTrainSampler, eval_windows
from model.ntu2p_stgcn import FID_JOINTS, XYZSTGCN, recognizer_input
from train.train_ntu2p_v2 import _device
from utils.fixseed import fixseed

NUM_ACTIONS = 26
# 项目已有的识别门槛（ReGenNet eval/action_xyz_classifier.py 的 GATE_THRESHOLDS）。
GATE_THRESHOLDS = OrderedDict([("top1_acc", 0.80), ("top5_acc", 0.90), ("balanced_acc", 0.55)])
SUBJVAL_MANIFEST = "results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json"
SUBJVAL_CACHE = "results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval"
ORIGINAL_MANIFEST = "results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"
ORIGINAL_CACHE = "results/forecasting/ntu120_label/ntu2p_xyz_seq_cache"


def classification_metrics(logits, labels, num_class=NUM_ACTIONS):
    labels = labels.long().cpu()
    logits = logits.float().cpu()
    top1 = logits.argmax(dim=-1)
    top5 = logits.topk(5, dim=-1).indices
    per_class, counts = [], []
    for k in range(int(num_class)):
        mask = labels == k
        counts.append(int(mask.sum().item()))
        per_class.append(float((top1[mask] == k).float().mean().item()) if counts[-1] > 0 else None)
    present = [value for value in per_class if value is not None]
    return OrderedDict(
        [
            ("top1_acc", float((top1 == labels).float().mean().item())),
            ("top5_acc", float((top5 == labels.unsqueeze(-1)).any(dim=-1).float().mean().item())),
            ("balanced_acc", float(sum(present) / len(present))),
            ("num_samples", int(labels.numel())),
            ("per_class_acc", per_class),
            ("per_class_count", counts),
        ]
    )


@torch.no_grad()
def predict(model, obs_xyz, future_xyz, device, batch_size=64):
    """逐批前向，返回 (logits, features)，均在 CPU、float64 特征。"""
    model.eval()
    logits, features = [], []
    for begin in range(0, int(obs_xyz.shape[0]), int(batch_size)):
        obs = obs_xyz[begin : begin + batch_size].to(device, dtype=torch.float32)
        future = future_xyz[begin : begin + batch_size].to(device, dtype=torch.float32)
        out, feat = model(recognizer_input(obs, future, model.joints))
        logits.append(out.cpu())
        features.append(feat.double().cpu())
    return torch.cat(logits), torch.cat(features)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def train(args):
    device = _device(args)
    fixseed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.save_dir, "recognizer.pt")) and not args.overwrite:
        raise FileExistsError("{} 已有 recognizer.pt，需 --overwrite".format(args.save_dir))
    _write_json(os.path.join(args.save_dir, "args.json"), vars(args))

    train_cache = NTU2PXYZSeqCache.load(args.cache_dir, "train", args.manifest_path, device=device)
    val_windows = eval_windows(NTU2PXYZSeqCache.load(args.cache_dir, "val", args.manifest_path, device="cpu"))
    sampler = NTU2PXYZTrainSampler(train_cache, args.batch_size, seed=args.seed)

    model = XYZSTGCN(num_class=NUM_ACTIONS, joints=FID_JOINTS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    open(log_path, "w").close()
    best = OrderedDict([("val_top1", -1.0), ("step", 0), ("state", None)])
    running = []
    for step in range(1, int(args.max_steps) + 1):
        model.train()
        batch = sampler.sample()
        logits, _ = model(recognizer_input(batch["obs_xyz"], batch["target_xyz"], model.joints))
        loss = F.cross_entropy(logits, batch["action"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running.append(float(loss.item()))
        if step % int(args.eval_interval) == 0 or step == int(args.max_steps):
            val_logits, _ = predict(model, val_windows["obs_xyz"], val_windows["target_xyz"], device)
            val = classification_metrics(val_logits, val_windows["action"])
            record = OrderedDict([("step", step), ("train_loss", sum(running) / len(running)),
                                  ("val_top1", val["top1_acc"]), ("val_top5", val["top5_acc"]),
                                  ("val_balanced", val["balanced_acc"]), ("epoch", sampler.epoch)])
            running = []
            if val["top1_acc"] > best["val_top1"]:
                best = OrderedDict([("val_top1", val["top1_acc"]), ("step", step),
                                    ("state", copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))])
            record["best_step"] = best["step"]
            with open(log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)

    model.load_state_dict(best["state"])
    checkpoint = os.path.join(args.save_dir, "recognizer.pt")
    torch.save(OrderedDict([("config", model.config()), ("model", best["state"]), ("step", best["step"]),
                            ("val_top1", best["val_top1"]), ("seed", int(args.seed))]), checkpoint)

    test_windows = eval_windows(NTU2PXYZSeqCache.load(args.test_cache_dir, "test", args.test_manifest_path, device="cpu"))
    test_logits, _ = predict(model, test_windows["obs_xyz"], test_windows["target_xyz"], device)
    test = classification_metrics(test_logits, test_windows["action"])
    gate = OrderedDict((key, test[key] >= threshold) for key, threshold in GATE_THRESHOLDS.items())
    summary = OrderedDict(
        [
            ("checkpoint", checkpoint),
            ("sha256", sha256(checkpoint)),
            ("best_step", best["step"]),
            ("val_top1", best["val_top1"]),
            ("test_gt", test),
            ("gate_thresholds", GATE_THRESHOLDS),
            ("gate", gate),
            ("passes_gate", all(gate.values())),
            ("created_at", datetime.utcnow().isoformat() + "Z"),
        ]
    )
    _write_json(os.path.join(args.save_dir, "metrics.json"), summary)
    print(json.dumps(OrderedDict((k, v) for k, v in summary.items() if k != "test_gt"), ensure_ascii=False), flush=True)
    print("test GT top1/top5/balanced: {top1_acc:.4f} / {top5_acc:.4f} / {balanced_acc:.4f}".format(**test), flush=True)
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--cache_dir", default=SUBJVAL_CACHE)
    parser.add_argument("--manifest_path", default=SUBJVAL_MANIFEST)
    parser.add_argument("--test_cache_dir", default=ORIGINAL_CACHE)
    parser.add_argument("--test_manifest_path", default=ORIGINAL_MANIFEST)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow_cpu_for_smoke_test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
