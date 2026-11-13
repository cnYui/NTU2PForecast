"""F13 补 seed 验证：按预登记判定（docs/ai/context/20261106-120018-ntu2p-f13-seed-verification-preregistration.md）。

判定只用新补的 seed1、seed2（seed0 是发现提升的 seed）；3 seed 均值另报。读出 ode0 + 落脚规划，参照同 seed 的 BR。
"""

import json
import statistics
from collections import OrderedDict

from scripts.run_ntu2p_jitflow import BASE
from scripts.run_ntu2p_legx import full_row

SUB = BASE / "ntu2p_jitflow" / "subjval"
CONFIRM, ALL = (1, 2), (0, 1, 2)


def main():
    data = OrderedDict([("BR", OrderedDict((s, full_row(SUB / "jointgraph_s{}".format(s), "BR")) for s in ALL)),
                        ("F13", OrderedDict((s, full_row(SUB / "partgraph_s{}".format(s), "F13")) for s in ALL))])
    m = lambda name, key, seeds=CONFIRM: statistics.mean(data[name][s][key] for s in seeds)  # noqa: E731
    checks = OrderedDict(
        [
            ("mpjpe ≤ 0.992×BR（seed1/2 均值）", m("F13", "mpjpe") <= 0.992 * m("BR", "mpjpe")),
            ("xyz_mse ≤ BR", m("F13", "xyz_mse") <= m("BR", "xyz_mse")),
            ("xyz_mae ≤ BR", m("F13", "xyz_mae") <= m("BR", "xyz_mae")),
            ("逐 seed mpjpe 都低于 BR", all(data["F13"][s]["mpjpe"] < data["BR"][s]["mpjpe"] for s in CONFIRM)),
            ("护栏 滑行 ≤ BR+0.01", m("F13", "slide_frame_ratio") <= m("BR", "slide_frame_ratio") + 0.01),
            ("护栏 步数 ≥ BR−0.1", m("F13", "step_count") >= m("BR", "step_count") - 0.1),
            ("护栏 jerk ≤ 24.5", m("F13", "jerk_body") <= 24.5),
            ("护栏 骨长 ≤ 1e-5", m("F13", "bone_rel_err_body") <= 1e-5),
            ("护栏 穿地 ≤ BR+0.01", m("F13", "foot_penetration_ratio") <= m("BR", "foot_penetration_ratio") + 0.01),
            ("护栏 摆动 gate", m("F13", "dct_low_ratio") >= 0.40 and m("F13", "dct_mid_ratio") >= 0.10),
            ("护栏 超额后退 ≤ 1.1×BR", m("F13", "primary_excess_antic") <= 1.1 * m("BR", "primary_excess_antic")),
            ("护栏 漏接触 ≤ 1.1×BR", m("F13", "missed_contact_total") <= 1.1 * m("BR", "missed_contact_total")),
            ("护栏 左右交替相关 ≥ BR−0.05", m("F13", "gait_sep_corr_f01_10") >= m("BR", "gait_sep_corr_f01_10") - 0.05),
        ]
    )
    keys = list(data["BR"][0])
    summary = OrderedDict([("per_seed", data),
                           ("confirm_mean", OrderedDict((n, OrderedDict((k, m(n, k)) for k in keys)) for n in data)),
                           ("all_mean", OrderedDict((n, OrderedDict((k, m(n, k, ALL)) for k in keys)) for n in data)),
                           ("checks", checks), ("pass", all(checks.values()))])
    (SUB / "f13_verification_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    show = ["mpjpe", "xyz_mse", "xyz_mae", "primary_excess_antic", "missed_contact_total", "wrong_overlap", "slide_frame_ratio",
            "step_count", "gait_sep_corr_f01_10", "jerk_body", "dct_mid_ratio"]
    lines = ["# F13 补 seed 验证（ode0 + 落脚规划，H-val）", "", "|行|" + "|".join(show) + "|", "|---|" + "---:|" * len(show)]
    for name in data:
        for s in ALL:
            lines.append("|{} s{}|{}|".format(name, s, "|".join("{:.4f}".format(data[name][s][k]) for k in show)))
    for label, block in (("seed1/2 均值（判定）", summary["confirm_mean"]), ("3 seed 均值（报告）", summary["all_mean"])):
        for name in data:
            lines.append("|{} {}|{}|".format(name, label, "|".join("{:.4f}".format(block[name][k]) for k in show)))
    d = lambda block, k: 100 * (block["F13"][k] / block["BR"][k] - 1)  # noqa: E731
    lines += ["", "F13 相对 BR：seed1/2 均值 mpjpe {:+.2f}%、xyz_mse {:+.2f}%、xyz_mae {:+.2f}%；3 seed 均值 mpjpe {:+.2f}%、xyz_mse {:+.2f}%".format(
        d(summary["confirm_mean"], "mpjpe"), d(summary["confirm_mean"], "xyz_mse"), d(summary["confirm_mean"], "xyz_mae"),
        d(summary["all_mean"], "mpjpe"), d(summary["all_mean"], "xyz_mse"))]
    lines += ["", "判定：" + ("通过" if summary["pass"] else "不通过")] + ["- {}：{}".format(k, "过" if v else "未过") for k, v in checks.items()]
    (SUB / "f13_verification_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
