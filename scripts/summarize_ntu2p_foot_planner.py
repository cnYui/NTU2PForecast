"""落脚规划层 H-val 3 seed 汇总与预登记判定（设计 20261104-170558 第 4 节，修订 20261104-171132）。

行：FD2、FD2 + 规划层、BR:ode0、BR:ode0 + 规划层，均为 eval_ntu2p_v2 的输出（bandflow_s<seed>/ 下）。
判定对象是 BR:ode0 + 规划层的 3 seed 均值，参照 BR:ode0 自身与 FD2。
"""

import json
import statistics
from collections import OrderedDict
from pathlib import Path

L = Path("results/forecasting/ntu120_label/ntu2p_jitflow/subjval")
ROWS = OrderedDict([("FD2", "FD2ref_endpoint.json"), ("FD2 + 规划层", "FD2ref_plan.json"),
                    ("BR:ode0", "BR_ode0.json"), ("BR:ode0 + 规划层", "BR_ode0plan.json")])
NAT = ("slide_frame_ratio", "lr_forward_vel_corr", "step_count", "skate_gt_contact_speed_walk", "gait_sep_corr_f01_10",
       "stance_fraction", "jerk_body", "foot_penetration_ratio", "bone_rel_err_body")
GT_STEPS = 2.288


def row(result):
    values = OrderedDict((key, result["model_metrics"][key]) for key in ("xyz_mse", "xyz_mae", "mpjpe"))
    natural = result["naturalness"]["model"]
    values.update((key, natural[key]) for key in NAT)
    values["dct_mid_ratio"] = result["articulation_gate"]["model_dct_mid_ratio"]
    values["primary_excess_antic"] = result["interaction_timing"]["primary_excess_antic"]["model"]
    return values


def main():
    per_seed = OrderedDict((name, [row(json.loads((L / "bandflow_s{}".format(s) / file).read_text())) for s in range(3)])
                           for name, file in ROWS.items())
    mean = OrderedDict((name, OrderedDict((k, statistics.mean(r[k] for r in rows)) for k in rows[0])) for name, rows in per_seed.items())
    std = OrderedDict((name, OrderedDict((k, statistics.stdev(r[k] for r in rows)) for k in rows[0])) for name, rows in per_seed.items())
    plan, base, fd2 = mean["BR:ode0 + 规划层"], mean["BR:ode0"], mean["FD2"]
    checks = OrderedDict([
        ("滑行帧比 ≤ 0.5×BR", plan["slide_frame_ratio"] <= 0.5 * base["slide_frame_ratio"]),
        ("左右交替相关 ≤ −0.25", plan["lr_forward_vel_corr"] <= -0.25),
        ("步数更接近 GT", abs(plan["step_count"] - GT_STEPS) < abs(base["step_count"] - GT_STEPS)),
        ("GT 着地帧脚速（步行）不高于 BR", plan["skate_gt_contact_speed_walk"] <= base["skate_gt_contact_speed_walk"]),
        ("护栏 mpjpe ≤ 1.01×BR", plan["mpjpe"] <= 1.01 * base["mpjpe"]),
        ("护栏 骨长 ≤ 1e-5", plan["bone_rel_err_body"] <= 1e-5),
        ("护栏 穿地 ≤ BR+0.01", plan["foot_penetration_ratio"] <= base["foot_penetration_ratio"] + 0.01),
        ("护栏 jerk ≤ 1.2×FD2", plan["jerk_body"] <= 1.2 * fd2["jerk_body"]),
    ])
    out = OrderedDict([("mean", mean), ("std", std), ("per_seed", per_seed), ("checks", checks), ("metric_pass", all(checks.values()))])
    (L / "bandflow_foot_planner_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    keys = ["mpjpe", "xyz_mse", "xyz_mae"] + list(NAT) + ["dct_mid_ratio", "primary_excess_antic"]
    lines = ["# 落脚规划层 H-val 3 seed", "", "|行|" + "|".join(keys) + "|", "|---|" + "---:|" * len(keys)]
    for name in ROWS:
        lines.append("|{}|{}|".format(name, "|".join("{:.4f}".format(mean[name][k]) for k in keys)))
    lines += ["", "判定（BR:ode0 + 规划层）："] + ["- {}：{}".format(k, "过" if v else "未过") for k, v in checks.items()]
    (L / "bandflow_foot_planner_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
