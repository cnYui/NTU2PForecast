"""G1 精度副发现的验证：按预登记判定（docs/ai/context/20261104-210308-ntu2p-g1-accuracy-verification-preregistration.md）。

判定只用新补的 seed1、seed2（seed0 是发现提升的 seed，计入会偏乐观）；3 seed 均值另报。
读出均为 ode0 + 落脚规划层，参照同 seed 的主线 BR。
"""

import json
import statistics
from collections import OrderedDict
from pathlib import Path

from scripts.run_ntu2p_jointgraph import row

BASE = Path("results/forecasting/ntu120_label/ntu2p_jitflow/subjval")
CONFIRM_SEEDS, ALL_SEEDS = (1, 2), (0, 1, 2)


def load(seed, name):
    out = BASE / "jointgraph_s{}".format(seed)
    values = row(out, name, "ode0plan")
    result = json.loads((out / "{}_ode0plan.json".format(name)).read_text())
    values["dct_low_ratio"] = result["articulation_gate"]["model_dct_low_ratio"]
    return values


def mean(rows, key):
    return statistics.mean(r[key] for r in rows)


def main():
    data = OrderedDict((name, OrderedDict((s, load(s, name)) for s in ALL_SEEDS)) for name in ("BR", "G1"))
    br = [data["BR"][s] for s in CONFIRM_SEEDS]
    g1 = [data["G1"][s] for s in CONFIRM_SEEDS]
    m = lambda rows, k: mean(rows, k)  # noqa: E731
    checks = OrderedDict(
        [
            ("mpjpe ≤ 0.992×BR（seed1/2 均值）", m(g1, "mpjpe") <= 0.992 * m(br, "mpjpe")),
            ("xyz_mse ≤ BR", m(g1, "xyz_mse") <= m(br, "xyz_mse")),
            ("xyz_mae ≤ BR", m(g1, "xyz_mae") <= m(br, "xyz_mae")),
            ("逐 seed mpjpe 都低于 BR", all(data["G1"][s]["mpjpe"] < data["BR"][s]["mpjpe"] for s in CONFIRM_SEEDS)),
            ("护栏 滑行 ≤ BR+0.01", m(g1, "slide_frame_ratio") <= m(br, "slide_frame_ratio") + 0.01),
            ("护栏 步数 ≥ BR−0.1", m(g1, "step_count") >= m(br, "step_count") - 0.1),
            ("护栏 jerk ≤ 24.5", m(g1, "jerk_body") <= 24.5),
            ("护栏 骨长 ≤ 1e-5", m(g1, "bone_rel_err_body") <= 1e-5),
            ("护栏 穿地 ≤ BR+0.01", m(g1, "foot_penetration_ratio") <= m(br, "foot_penetration_ratio") + 0.01),
            ("护栏 摆动 gate", m(g1, "dct_low_ratio") >= 0.40 and m(g1, "dct_mid_ratio") >= 0.10),
            ("护栏 超额后退 ≤ 1.1×BR", m(g1, "primary_excess_antic") <= 1.1 * m(br, "primary_excess_antic")),
            ("护栏 漏接触 ≤ 1.1×BR", m(g1, "missed_contact_total") <= 1.1 * m(br, "missed_contact_total")),
        ]
    )
    keys = list(next(iter(data["BR"].values())))
    summary = OrderedDict(
        [
            ("per_seed", data),
            ("confirm_mean", OrderedDict((n, OrderedDict((k, m([data[n][s] for s in CONFIRM_SEEDS], k)) for k in keys)) for n in data)),
            ("all_mean", OrderedDict((n, OrderedDict((k, m([data[n][s] for s in ALL_SEEDS], k)) for k in keys)) for n in data)),
            ("checks", checks),
            ("metric_pass", all(checks.values())),
        ]
    )
    (BASE / "g1_verification_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    show = ["mpjpe", "xyz_mse", "xyz_mae", "primary_excess_antic", "missed_contact_total", "wrong_overlap", "slide_frame_ratio",
            "step_count", "jerk_body", "dct_mid_ratio"]
    lines = ["# G1 精度验证（ode0 + 落脚规划，H-val）", "", "|行|" + "|".join(show) + "|", "|---|" + "---:|" * len(show)]
    for name in data:
        for s in ALL_SEEDS:
            lines.append("|{} s{}|{}|".format(name, s, "|".join("{:.4f}".format(data[name][s][k]) for k in show)))
    for label, block in (("seed1/2 均值（判定用）", summary["confirm_mean"]), ("3 seed 均值（只报告）", summary["all_mean"])):
        for name in data:
            lines.append("|{} {}|{}|".format(name, label, "|".join("{:.4f}".format(block[name][k]) for k in show)))
    delta = lambda block, k: 100 * (block["G1"][k] / block["BR"][k] - 1)  # noqa: E731
    lines += ["", "G1 相对 BR：seed1/2 均值 mpjpe {:+.2f}%、xyz_mse {:+.2f}%、xyz_mae {:+.2f}%；3 seed 均值 mpjpe {:+.2f}%、xyz_mse {:+.2f}%".format(
        delta(summary["confirm_mean"], "mpjpe"), delta(summary["confirm_mean"], "xyz_mse"), delta(summary["confirm_mean"], "xyz_mae"),
        delta(summary["all_mean"], "mpjpe"), delta(summary["all_mean"], "xyz_mse"))]
    lines += ["", "判定：" + ("通过" if summary["metric_pass"] else "不通过")] + ["- {}：{}".format(k, "过" if v else "未过") for k, v in checks.items()]
    (BASE / "g1_verification_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
