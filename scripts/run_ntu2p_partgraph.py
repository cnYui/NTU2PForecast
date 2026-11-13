"""细分部位（F13）与骨架邻接图（SK）两个单因子对照：H-val 训练、评价与预登记判定。

预登记：docs/ai/context/20261106-091321-ntu2p-fine-parts-and-skeleton-graph-preregistration.md。
判定规则与 XL 相同（复用 run_ntu2p_legx.decide）；参照为同 seed 的 BR（jointgraph_s<seed>/ 已评过的 ode0 + 落脚规划）。
"""

import argparse
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from scripts.run_ntu2p_bandflow import CANDIDATES, FLOW_BASE, eval_readout, train_candidate
from scripts.run_ntu2p_jitflow import BASE, PROTOCOLS, say, score_row
from scripts.run_ntu2p_jointgraph import PLANNER
from scripts.run_ntu2p_legx import decide, full_row

RECIPE = ["--flow_recipe_weight", "1.0", "--flow_recipe_t_min", "0.5"]
CANDIDATES["F13"] = FLOW_BASE + ["--flow_kwargs", json.dumps({"local_k": 11, "part_scheme": "fine"})] + RECIPE
CANDIDATES["SK"] = FLOW_BASE + ["--flow_kwargs", json.dumps({"local_k": 11, "skeleton_mask": True})] + RECIPE
NAMES = ("F13", "SK")


def job(opts, name, out):
    protocol = PROTOCOLS["subjval"]
    log = out / "{}.log".format(name)
    checkpoint = train_candidate(opts, name, opts.seed, protocol, log)
    say("评价 {}".format(name))
    eval_readout(checkpoint, out / "{}_ode0plan.json".format(name), protocol, log, "sample", 0.0,
                 export=out / "{}_ode0plan_arrays.pt".format(name), extra=PLANNER)
    eval_readout(checkpoint, out / "{}_ode0.json".format(name), protocol, log, "sample", 0.0)
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--names", nargs="+", default=list(NAMES), choices=NAMES)
    opts = parser.parse_args()
    opts.protocol = "subjval"
    sub = BASE / "ntu2p_jitflow" / "subjval"
    out = sub / "partgraph_s{}".format(opts.seed)
    out.mkdir(parents=True, exist_ok=True)
    names = tuple(opts.names)
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        for future in [pool.submit(job, opts, name, out) for name in names]:
            future.result()
    rows = OrderedDict([("BR", full_row(sub / "jointgraph_s{}".format(opts.seed), "BR"))])
    rows.update((name, full_row(out, name)) for name in names)
    ode0 = OrderedDict([("BR", score_row(json.loads((sub / "bandflow_s{}".format(opts.seed) / "BR_ode0.json").read_text()))["mpjpe"])])
    ode0.update((name, score_row(json.loads((out / "{}_ode0.json".format(name)).read_text()))["mpjpe"]) for name in names)
    decisions = OrderedDict()
    for name in names:
        gains, guards, passed = decide(rows[name], rows["BR"])
        decisions[name] = OrderedDict([("gains", gains), ("guards", guards), ("pass", passed)])
    (out / "summary.json").write_text(json.dumps(OrderedDict([("ode0plan", rows), ("ode0_mpjpe", ode0), ("decisions", decisions)]),
                                                 ensure_ascii=False, indent=2))
    keys = list(rows["BR"])
    lines = ["# 细分部位 F13 / 骨架邻接图 SK vs BR（ode0 + 落脚规划，H-val seed{}）".format(opts.seed), "",
             "|行|" + "|".join(keys) + "|", "|---|" + "---:|" * len(keys)]
    for name, values in rows.items():
        lines.append("|{}|{}|".format(name, "|".join("{:.4f}".format(v) for v in values.values())))
    lines += ["", "ode0（不加规划）mpjpe：" + "，".join("{} {:.5f}".format(k, v) for k, v in ode0.items())]
    for name, d in decisions.items():
        lines += ["", "{} 判定：{}".format(name, "有增量" if d["pass"] else "无增量")]
        lines += ["- 增量 {}：{}".format(k, "是" if v else "否") for k, v in d["gains"].items()]
        lines += ["- 护栏 {}：{}".format(k, "过" if v else "未过") for k, v in d["guards"].items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
