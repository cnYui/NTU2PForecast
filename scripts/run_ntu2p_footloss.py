"""把落脚规划融入训练：第 1 轮 FS（加强 GT 锁脚）/ FC（着地概率头 + 自洽锁脚）；第 2 轮 FH（落脚项只在高 t 分支）/
FHS（再加摆动监督）。训练、评价与预登记判定。

预登记：docs/ai/context/20261106-185013-ntu2p-foot-planting-in-training-design-and-preregistration.md（第 1 轮）、
20261106-195544-ntu2p-foot-planting-v2-preregistration.md（第 2 轮，判定规则相同）。
评价一律 ode0、不加落脚规划；参照 BR 不加规划（同命令重评）与 BR + 规划（jointgraph_s<seed>/ 已有）。
"""

import argparse
import json
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.run_ntu2p_bandflow import CANDIDATES, FLOW_BASE, eval_readout, train_candidate
from scripts.run_ntu2p_jitflow import BASE, PROTOCOLS, say
from scripts.run_ntu2p_jointgraph import PLANNER, row

RECIPE = ["--flow_recipe_weight", "1.0", "--flow_recipe_t_min", "0.5"]
CANDIDATES["FS"] = FLOW_BASE + ["--flow_kwargs", json.dumps({"local_k": 11})] + RECIPE + ["--foot_loss_weight", "1.0"]
CANDIDATES["FC"] = FLOW_BASE + ["--flow_kwargs", json.dumps({"local_k": 11, "contact_head": True})] + RECIPE + [
    "--foot_contact_bce_weight", "0.1", "--foot_self_lock_weight", "1.0"]
HIGH_T = ["--foot_terms_branch", "flow"]
CANDIDATES["FH"] = CANDIDATES["FC"] + HIGH_T
CANDIDATES["FHS"] = CANDIDATES["FC"] + HIGH_T + ["--foot_swing_weight", "0.5"]
ROUNDS = {1: ("FS", "FC"), 2: ("FH", "FHS")}


def full_row(out, name, readout):
    values = row(out, name, readout)
    result = json.loads((out / "{}_{}.json".format(name, readout)).read_text())
    values["gait_sep_corr_f01_10"] = result["naturalness"]["model"].get("gait_sep_corr_f01_10")
    values["dct_low_ratio"] = result["articulation_gate"]["model_dct_low_ratio"]
    return values


def gait_level(cand, ref):
    """不看护栏时的步态等级（None 为无效）；随机读出轮要单独报告"步态达标、精度未过"。"""
    if cand["slide_frame_ratio"] <= 0.06 and cand["step_count"] >= 1.60:
        return "可替代落脚规划层"
    if cand["slide_frame_ratio"] <= 0.10 and cand["step_count"] >= ref["step_count"]:
        return "部分有效"
    return None


def decide(cand, ref, mpjpe_tol=1.01):
    guards = OrderedDict(
        [
            ("mpjpe ≤ {}×BR".format(mpjpe_tol), cand["mpjpe"] <= mpjpe_tol * ref["mpjpe"]),
            ("xyz_mse ≤ 1.02×BR", cand["xyz_mse"] <= 1.02 * ref["xyz_mse"]),
            ("jerk ≤ 24.5", cand["jerk_body"] <= 24.5),
            ("穿地 ≤ BR+0.01", cand["foot_penetration_ratio"] <= ref["foot_penetration_ratio"] + 0.01),
            ("骨长 ≤ 1e-5", cand["bone_rel_err_body"] <= 1e-5),
            ("摆动 gate", cand["dct_low_ratio"] >= 0.40 and cand["dct_mid_ratio"] >= 0.10),
            ("超额后退 ≤ 1.1×BR", cand["primary_excess_antic"] <= 1.1 * ref["primary_excess_antic"]),
            ("漏接触 ≤ 1.1×BR", cand["missed_contact_total"] <= 1.1 * ref["missed_contact_total"]),
        ]
    )
    level = "无效（护栏未过）" if not all(guards.values()) else gait_level(cand, ref) or "无效"
    return guards, level


def job(opts, name, out):
    protocol = PROTOCOLS["subjval"]
    log = out / "{}.log".format(name)
    if name == "BR":
        checkpoint = Path("save/forecasting/ntu120_label/ntu2p_bandflow_subjval_BR_s{}_10000/ema/model000010000.pt".format(opts.seed))
    else:
        checkpoint = train_candidate(opts, name, opts.seed, protocol, log)
    say("评价 {}".format(name))
    eval_readout(checkpoint, out / "{}_ode0.json".format(name), protocol, log, "sample", 0.0, export=out / "{}_ode0_arrays.pt".format(name))
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    opts = parser.parse_args()
    opts.protocol = "subjval"
    sub = BASE / "ntu2p_jitflow" / "subjval"
    names = ROUNDS[opts.round]
    out = sub / ("footloss_s{}" if opts.round == 1 else "footloss{}_s{{}}".format(opts.round)).format(opts.seed)
    out.mkdir(parents=True, exist_ok=True)
    first = sub / "footloss_s{}".format(opts.seed)
    for suffix in ("_ode0.json", "_ode0_arrays.pt"):
        # BR 不加规划的评价与第 1 轮同一命令，已有则复用，避免重复评价。
        if opts.round > 1 and (first / ("BR" + suffix)).exists() and not (out / ("BR" + suffix)).exists():
            shutil.copyfile(first / ("BR" + suffix), out / ("BR" + suffix))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(job, opts, name, out) for name in ("BR",) + names]:
            future.result()
    rows = OrderedDict([("BR", full_row(out, "BR", "ode0"))])
    rows.update((name, full_row(out, name, "ode0")) for name in names)
    rows["BR+规划"] = full_row(sub / "jointgraph_s{}".format(opts.seed), "BR", "ode0plan")
    decisions = OrderedDict()
    for name in names:
        guards, level = decide(rows[name], rows["BR"])
        decisions[name] = OrderedDict([("guards", guards), ("level", level)])
    (out / "summary.json").write_text(json.dumps(OrderedDict([("rows", rows), ("decisions", decisions)]), ensure_ascii=False, indent=2))
    keys = list(rows["BR"])
    lines = ["# 落脚融入训练第 {} 轮：{}（ode0 不加规划，H-val seed{}）".format(opts.round, " / ".join(names), opts.seed), "",
             "|行|" + "|".join(keys) + "|", "|---|" + "---:|" * len(keys)]
    for name, values in rows.items():
        lines.append("|{}|{}|".format(name, "|".join("{:.4f}".format(v) for v in values.values())))
    for name, d in decisions.items():
        lines += ["", "{} 判定：{}".format(name, d["level"])]
        lines += ["- 护栏 {}：{}".format(k, "过" if v else "未过") for k, v in d["guards"].items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
