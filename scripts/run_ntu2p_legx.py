"""跨人腿部时间 Transformer（XL）：H-val 训练、读出评价与预登记判定。

设计与预登记：docs/ai/context/20261105-221250-ntu2p-cross-person-leg-transformer-design-and-preregistration.md。
XL = 主线 BR + 零初始化的 LegCrossTransformer；参照为同 seed 的 BR（ode0 + 落脚规划已在 jointgraph_s<seed>/ 评过）。
"""

import argparse
import json
from collections import OrderedDict

from scripts.run_ntu2p_bandflow import CANDIDATES, FLOW_BASE, eval_readout, train_candidate
from scripts.run_ntu2p_jitflow import BASE, PROTOCOLS, say, score_row
from scripts.run_ntu2p_jointgraph import PLANNER, row

CANDIDATES["XL"] = FLOW_BASE + ["--flow_kwargs", json.dumps({"local_k": 11, "leg_xattn": True}),
                                "--flow_recipe_weight", "1.0", "--flow_recipe_t_min", "0.5"]


def full_row(out, name):
    values = row(out, name, "ode0plan")
    result = json.loads((out / "{}_ode0plan.json".format(name)).read_text())
    values["gait_sep_corr_f01_10"] = result["naturalness"]["model"].get("gait_sep_corr_f01_10")
    values["dct_low_ratio"] = result["articulation_gate"]["model_dct_low_ratio"]
    return values


def decide(cand, ref):
    gains = OrderedDict(
        [
            ("精度：mpjpe ≤ 0.992×BR 且 xyz_mse ≤ BR", cand["mpjpe"] <= 0.992 * ref["mpjpe"] and cand["xyz_mse"] <= ref["xyz_mse"]),
            ("交互：超额后退 ≤ 0.9×BR 且漏接触 ≤ BR",
             cand["primary_excess_antic"] <= 0.9 * ref["primary_excess_antic"] and cand["missed_contact_total"] <= ref["missed_contact_total"]),
            ("腿：滑行 ≤ BR−0.005、步数 ≥ BR、交替相关 ≥ BR",
             cand["slide_frame_ratio"] <= ref["slide_frame_ratio"] - 0.005 and cand["step_count"] >= ref["step_count"]
             and cand["gait_sep_corr_f01_10"] >= ref["gait_sep_corr_f01_10"]),
        ]
    )
    guards = OrderedDict(
        [
            ("mpjpe ≤ 1.01×BR", cand["mpjpe"] <= 1.01 * ref["mpjpe"]),
            ("jerk ≤ 24.5", cand["jerk_body"] <= 24.5),
            ("骨长 ≤ 1e-5", cand["bone_rel_err_body"] <= 1e-5),
            ("穿地 ≤ BR+0.01", cand["foot_penetration_ratio"] <= ref["foot_penetration_ratio"] + 0.01),
            ("摆动 gate", cand["dct_low_ratio"] >= 0.40 and cand["dct_mid_ratio"] >= 0.10),
        ]
    )
    return gains, guards, any(gains.values()) and all(guards.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    opts = parser.parse_args()
    opts.protocol = "subjval"
    protocol = PROTOCOLS["subjval"]
    out = BASE / "ntu2p_jitflow" / "subjval" / "legx_s{}".format(opts.seed)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "XL.log"
    checkpoint = train_candidate(opts, "XL", opts.seed, protocol, log)
    say("评价 XL")
    eval_readout(checkpoint, out / "XL_ode0plan.json", protocol, log, "sample", 0.0, export=out / "XL_ode0plan_arrays.pt", extra=PLANNER)
    eval_readout(checkpoint, out / "XL_ode0.json", protocol, log, "sample", 0.0)
    ref_dir = BASE / "ntu2p_jitflow" / "subjval" / "jointgraph_s{}".format(opts.seed)
    rows = OrderedDict([("BR", full_row(ref_dir, "BR")), ("XL", full_row(out, "XL"))])
    gains, guards, passed = decide(rows["XL"], rows["BR"])
    ode0 = OrderedDict((name, score_row(json.loads(path.read_text()))) for name, path in
                       (("BR", BASE / "ntu2p_jitflow" / "subjval" / "bandflow_s{}".format(opts.seed) / "BR_ode0.json"), ("XL", out / "XL_ode0.json")))
    summary = OrderedDict([("ode0plan", rows), ("ode0_mpjpe", {k: v["mpjpe"] for k, v in ode0.items()}),
                           ("gains", gains), ("guards", guards), ("pass", passed)])
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    keys = list(rows["BR"])
    lines = ["# 跨人腿部 Transformer XL vs BR（ode0 + 落脚规划，H-val seed{}）".format(opts.seed), "",
             "|行|" + "|".join(keys) + "|", "|---|" + "---:|" * len(keys)]
    for name, values in rows.items():
        lines.append("|{}|{}|".format(name, "|".join("{:.4f}".format(v) for v in values.values())))
    lines += ["", "ode0（不加规划）mpjpe：BR {:.5f}，XL {:.5f}".format(ode0["BR"]["mpjpe"], ode0["XL"]["mpjpe"]), "",
              "判定：" + ("有增量" if passed else "无增量")]
    lines += ["- 增量 {}：{}".format(k, "是" if v else "否") for k, v in gains.items()]
    lines += ["- 护栏 {}：{}".format(k, "过" if v else "未过") for k, v in guards.items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
