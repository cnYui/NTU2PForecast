"""PGait v4b 原协议 test：原协议 train 上训练步态模块 3 seed，挂在原协议 BR:ode0 上每 seed 只评一次 test。

计划：docs/ai/context/20261107-110135-ntu2p-phase-gait-v4b-original-protocol-test-plan.md。
对照 BR:ode0 与 FD2 用主线 test 已有的同命令结果（original/bandflow_test/），不重评。
"""

import json
import math
import subprocess
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.run_ntu2p_bandflow import eval_readout
from scripts.run_ntu2p_footloss import decide, full_row
from scripts.run_ntu2p_footloss_sample import seed_row
from scripts.run_ntu2p_jitflow import BASE, PROTOCOLS, PYTHON, SAVE, run, say

V4B_ARGS = ["--ground_aware", "--penetration_weight", "10", "--obs_recon_weight", "1.0", "--f_steps", "12000", "--obs_pose_input",
            "--anchor_tau", "4", "--vel_lowpass_order", "8", "--leg_l2_weight", "3.0"]
SEEDS = (0, 1, 2)
MPJPE_TOL = 1.005


def job(seed, out):
    protocol = PROTOCOLS["original"]
    log = out / "s{}.log".format(seed)
    directory = SAVE / "ntu2p_phase_gait_v4b_original_s{}".format(seed)
    checkpoint = directory / "phase_gait.pt"
    if not checkpoint.exists():
        say("训练步态模块 s{}".format(seed))
        run([PYTHON, "train/train_ntu2p_phase_gait.py", "--save_dir", str(directory), "--seed", str(seed), "--device", "cuda:0",
             "--cache_dir", protocol["cache"], "--manifest_path", protocol["manifest"]] + V4B_ARGS, log)
    br = SAVE / "ntu2p_bandflow_original_BR_s{}_10000/ema/model000010000.pt".format(seed)
    say("test 评价 s{}".format(seed))
    eval_readout(br, out / "PG_ode0_s{}.json".format(seed), protocol, log, "sample", 0.0, split="test",
                 export=out / "PG_ode0_s{}_arrays.pt".format(seed), extra=["--gait_head", str(checkpoint)])
    return seed


def stats(rows):
    keys = list(rows[0])
    mean = OrderedDict((k, sum(r[k] for r in rows) / len(rows)) for k in keys)
    std = OrderedDict((k, math.sqrt(sum((r[k] - mean[k]) ** 2 for r in rows) / (len(rows) - 1))) for k in keys)
    return mean, std


def main():
    sub = BASE / "ntu2p_jitflow" / "original"
    out = sub / "phase_gait_test"
    ref = sub / "bandflow_test"
    out.mkdir(parents=True, exist_ok=True)
    # 训练每 seed 约 10 分钟、显存小，三者并行；test 评价随各自训练结束进行。
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(job, seed, out) for seed in SEEDS]:
            future.result()
    per_seed = OrderedDict()
    for seed in SEEDS:
        per_seed[seed] = OrderedDict([("BR:ode0", seed_row(ref, "BR", "ode0_s{}".format(seed))),
                                      ("PGait", seed_row(out, "PG", "ode0_s{}".format(seed))),
                                      ("FD2", full_row(ref, "FD2", "s{}".format(seed)))])
    rows = OrderedDict()
    for name in ("FD2", "BR:ode0", "PGait"):
        rows[name] = stats([per_seed[s][name] for s in SEEDS])
    guards, level = decide(rows["PGait"][0], rows["BR:ode0"][0], MPJPE_TOL)
    deltas = OrderedDict((s, 100 * (per_seed[s]["PGait"]["mpjpe"] / per_seed[s]["BR:ode0"]["mpjpe"] - 1)) for s in SEEDS)
    summary = OrderedDict([("per_seed", per_seed), ("mean", OrderedDict((k, v[0]) for k, v in rows.items())),
                           ("std", OrderedDict((k, v[1]) for k, v in rows.items())), ("mpjpe_delta_pct_vs_BR", deltas),
                           ("guards", guards), ("level", level)])
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    keys = ["mpjpe", "xyz_mse", "xyz_mae", "slide_frame_ratio", "step_count", "lr_forward_vel_corr", "stance_fraction",
            "foot_penetration_ratio", "jerk_body", "dct_mid_ratio", "primary_excess_antic", "missed_contact_total"]
    lines = ["# PGait v4b 原协议 test（1253 条，3 seed 均值 ± 标准差）", "", "|行|" + "|".join(keys) + "|", "|---|" + "---:|" * len(keys)]
    for name, (mean, std) in rows.items():
        cells = ["{:.4f} ± {:.4f}".format(mean[k], std[k]) if k in std else "—" for k in keys]
        lines.append("|{}|{}|".format(name, "|".join(cells)))
    lines += ["", "PGait 相对 BR:ode0 的 mpjpe（逐 seed）：" + " / ".join("{:+.2f}%".format(v) for v in deltas.values()),
              "", "原预登记判定（test 只报告）：" + level] + ["- 护栏 {}：{}".format(k, "过" if v else "未过") for k, v in guards.items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
