"""对照实验：不做 DCT 截断（root/局部均 49 阶 ≈ 直接在 xyz 残差序列上扩散）与主线 BR 的比较。

预登记：docs/ai/context/20261105-100258-ntu2p-fullband-diffusion-control-preregistration.md。
复用 run_ntu2p_bandflow 的训练/评价；BR 已有权重只补评价；输出 subjval/fullband_s<seed>/。
"""

import argparse
import json
from collections import OrderedDict

from scripts.run_ntu2p_bandflow import CANDIDATES, FLOW_BASE, eval_readout, run_dir, train_candidate
from scripts.run_ntu2p_jitflow import BASE, PROTOCOLS, say, score_row

CANDIDATES["BRfull"] = FLOW_BASE + ["--flow_kwargs", json.dumps({"root_k": 49, "local_k": 49}),
                                    "--flow_recipe_weight", "1.0", "--flow_recipe_t_min", "0.5"]
# 名称 -> (读出, 温度, 额外参数, 是否导出数组)
READOUTS = OrderedDict(
    [
        ("ode0", ("sample", 0.0, (), False)),
        ("ode0plan", ("sample", 0.0, ("--foot_planner",), True)),
        ("sample", ("sample", 1.0, (), True)),
    ]
)
KEYS = ("xyz_mse", "xyz_mae", "mpjpe", "jerk_body", "dct_high_ratio", "dct_mid_ratio", "slide_frame_ratio", "step_count",
        "foot_penetration_ratio")


def decide(rows):
    br, full = rows["BR"], rows["BRfull"]
    ratio = lambda readout, key: full[readout][key] / br[readout][key]  # noqa: E731
    worse = (ratio("ode0", "mpjpe") >= 1.008 or ratio("ode0", "jerk_body") >= 1.5 or ratio("sample", "jerk_body") >= 1.5
             or ratio("sample", "dct_high_ratio") >= 3.0)
    fine = (ratio("ode0", "mpjpe") <= 1.008 and ratio("ode0", "jerk_body") <= 1.2 and ratio("sample", "jerk_body") <= 1.2
            and ratio("sample", "dct_high_ratio") <= 1.5)
    return "全频更差" if worse else ("全频不差" if fine else "混合")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    opts = parser.parse_args()
    opts.protocol = "subjval"
    protocol = PROTOCOLS[opts.protocol]
    out = BASE / "ntu2p_jitflow" / opts.protocol / "fullband_s{}".format(opts.seed)
    out.mkdir(parents=True, exist_ok=True)
    rows = OrderedDict()
    for name in ("BRfull", "BR"):
        log = out / "{}.log".format(name)
        checkpoint = train_candidate(opts, name, opts.seed, protocol, log) if name == "BRfull" else run_dir(opts, "BR") / "ema/model{:09d}.pt".format(opts.steps)
        say("评价 {}".format(name))
        rows[name] = OrderedDict()
        for readout, (flow_readout, temperature, extra, export) in READOUTS.items():
            result = eval_readout(checkpoint, out / "{}_{}.json".format(name, readout), protocol, log, flow_readout, temperature,
                                  export=out / "{}_{}_arrays.pt".format(name, readout) if export else None, extra=extra)
            row = score_row(result)
            rows[name][readout] = OrderedDict((key, row[key]) for key in KEYS)
    verdict = decide(rows)
    (out / "summary.json").write_text(json.dumps(OrderedDict([("rows", rows), ("verdict", verdict)]), ensure_ascii=False, indent=2))
    lines = ["# 全频（49 阶）vs 主线 BR（8/11 阶），H-val seed{}".format(opts.seed), "", "|行|" + "|".join(KEYS) + "|",
             "|---|" + "---:|" * len(KEYS)]
    for name, by_readout in rows.items():
        for readout, row in by_readout.items():
            lines.append("|{}:{}|{}|".format(name, readout, "|".join("{:.5f}".format(row[key]) for key in KEYS)))
    lines += ["", "判定：" + verdict]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
