"""随机读出 + 落脚损失：已有 BR/FS/FC/FH/FHS（H-val）权重做单样本 τ 读出，不加落脚规划，按预登记判定。

预登记：docs/ai/context/20261107-090909-ntu2p-foot-planting-stochastic-readout-preregistration.md。
判定用 3 个采样种子的均值；护栏基准为 BR ode0 不加规划（footloss_s<seed>/ 已有），目标为 BR ode0 + 规划（jointgraph_s<seed>/）。
"""

import argparse
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from scripts.run_ntu2p_bandflow import eval_readout
from scripts.run_ntu2p_footloss import decide, full_row, gait_level
from scripts.run_ntu2p_jitflow import BASE, PROTOCOLS, say
from utils.ntu2p_naturalness import aggregate_stats, compute_naturalness_stats

MODELS = ("BR", "FS", "FC", "FH", "FHS")
TEMPERATURES = (0.3, 0.5)
SAMPLE_SEEDS = (2718, 2719, 2720)
PRECISION_GUARDS = ("mpjpe ≤ 1.01×BR", "xyz_mse ≤ 1.02×BR")
GAIT_KEYS = ("stance_fraction", "lr_forward_vel_corr", "skate_gt_contact_speed_walk")
SPREAD_KEYS = ("mpjpe", "xyz_mse", "slide_frame_ratio", "step_count")


def tag(temperature, sample_seed):
    return "t{:02d}s{}".format(int(round(temperature * 10)), sample_seed)


def checkpoint(name, seed):
    return Path("save/forecasting/ntu120_label/ntu2p_bandflow_subjval_{}_s{}_10000/ema/model000010000.pt".format(name, seed))


def job(opts, out, name, temperature, sample_seed):
    readout = tag(temperature, sample_seed)
    stem = out / "{}_{}".format(name, readout)
    eval_readout(checkpoint(name, opts.seed), Path(str(stem) + ".json"), PROTOCOLS["subjval"], Path(str(stem) + ".log"), "sample",
                 temperature, export=Path(str(stem) + "_arrays.pt"), extra=["--flow_sample_seed", str(sample_seed)])
    return name, readout


def gait_row(arrays_path):
    """站立占比等步态细项不在评价 JSON 里，从导出数组按 utils/ntu2p_naturalness 同口径补算。"""
    arrays = torch.load(str(arrays_path))
    stats = aggregate_stats(compute_naturalness_stats(arrays["methods"]["model"].double(), arrays["target_xyz"].double(), arrays["obs_xyz"].double()))
    return OrderedDict((key, float(stats[key])) for key in GAIT_KEYS)


def seed_row(directory, name, readout):
    values = full_row(directory, name, readout)
    values.update(gait_row(directory / "{}_{}_arrays.pt".format(name, readout)))
    return values


def mean_row(per_seed):
    keys = list(per_seed[0])
    mean = OrderedDict((key, sum(row[key] for row in per_seed) / len(per_seed)) for key in keys)
    spread = OrderedDict((key, (min(row[key] for row in per_seed), max(row[key] for row in per_seed))) for key in SPREAD_KEYS)
    return mean, spread


def judge(cand, ref):
    guards, level = decide(cand, ref)
    others_ok = all(ok for key, ok in guards.items() if key not in PRECISION_GUARDS)
    gait = gait_level(cand, ref)
    if level == "无效（护栏未过）" and others_ok and gait:
        # 预登记：只差精度护栏时单独标出，是否接受精度代价由用户决定，不自动采纳。
        level = "步态达标（{}级）、精度未过".format(gait)
    return guards, level


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    opts = parser.parse_args()
    sub = BASE / "ntu2p_jitflow" / "subjval"
    out = sub / "footloss_sample_s{}".format(opts.seed)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(name, t, s) for t in TEMPERATURES for name in MODELS for s in SAMPLE_SEEDS]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(job, opts, out, *task) for task in tasks]:
            name, readout = future.result()
            say("评完 {} {}".format(name, readout))
    rows, spreads = OrderedDict(), OrderedDict()
    for directory, name, readout, label in ((sub / "footloss_s{}".format(opts.seed), "BR", "ode0", "BR:ode0"),
                                            (sub / "jointgraph_s{}".format(opts.seed), "BR", "ode0plan", "BR:ode0+规划")):
        rows[label] = seed_row(directory, name, readout)
    for temperature in TEMPERATURES:
        for name in MODELS:
            label = "{}:τ{}".format(name, temperature)
            rows[label], spreads[label] = mean_row([seed_row(out, name, tag(temperature, s)) for s in SAMPLE_SEEDS])
    ref = rows["BR:ode0"]
    decisions = OrderedDict()
    for label in spreads:
        guards, level = judge(rows[label], ref)
        name, temperature = label.split(":")
        control = rows["BR:" + temperature]
        # 归因：步数与滑行都优于同 τ 的 BR，才算落脚损失的作用。
        loss_helps = None if name == "BR" else (rows[label]["step_count"] > control["step_count"]
                                                and rows[label]["slide_frame_ratio"] < control["slide_frame_ratio"])
        decisions[label] = OrderedDict([("guards", guards), ("level", level), ("loss_helps_vs_BR_same_tau", loss_helps)])
    summary = OrderedDict([("rows", rows), ("spreads", spreads), ("decisions", decisions)])
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    keys = list(ref)
    lines = ["# 随机读出 + 落脚损失（单样本、不加规划，H-val seed{}，3 采样种子均值）".format(opts.seed), "",
             "|行|" + "|".join(keys) + "|", "|---|" + "---:|" * len(keys)]
    lines += ["|{}|{}|".format(label, "|".join("{:.4f}".format(v) for v in values.values())) for label, values in rows.items()]
    lines += ["", "## 种子范围（min–max）", ""]
    lines += ["- {}：{}".format(label, "，".join("{} {:.4f}–{:.4f}".format(k, *v) for k, v in spread.items())) for label, spread in spreads.items()]
    for label, d in decisions.items():
        lines += ["", "{} 判定：{}；落脚损失相对同 τ BR 有增益：{}".format(label, d["level"], d["loss_helps_vs_BR_same_tau"])]
        lines += ["- 护栏 {}：{}".format(k, "过" if v else "未过") for k, v in d["guards"].items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
