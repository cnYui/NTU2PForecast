"""扩散完整预测（带限流 + 采样读出）：训练、val 读出评价与预登记判定。

设计与预登记：docs/ai/context/20261103-215324-ntu2p-diffusion-full-prediction-design-and-plan.md。
候选：J3（上一轮 STG 流，已有权重，只补新读出）、B（局部 DCT 11 阶）、BR（B + 高 t 配方分支）。
每个候选评价三种可采纳读出（ode0 / median16 / median16t05，全部为 Heun 20 步多步采样）与两种参照读出
（endpoint、τ=1 单样本）；判定复用 run_ntu2p_jitflow 的规则，参照为同 seed 的 FD2 H-val EMA 10000 终点。
输出写在 ntu2p_jitflow/<protocol>/bandflow_s<seed>/，跨 seed 汇总直接用 scripts/summarize_ntu2p_jitflow_seeds.py。
"""

import argparse
import json
import shutil
import statistics
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.run_ntu2p_jitflow import (
    BASE,
    FD2_ADDONS,
    FD2_CONFIG,
    PROTOCOLS,
    PYTHON,
    S2_5_RECIPE,
    SAVE,
    common_args,
    decide,
    run,
    sample_row,
    say,
    score_row,
)


FLOW_BASE = ["--denoiser", "stg", "--self_cond", "--t_uniform_mix", "0.5"]
CANDIDATES = OrderedDict(
    [
        ("B", FLOW_BASE + ["--flow_kwargs", json.dumps({"local_k": 11})]),
        ("BR", FLOW_BASE + ["--flow_kwargs", json.dumps({"local_k": 11}), "--flow_recipe_weight", "1.0", "--flow_recipe_t_min", "0.5"]),
        # 关节级交互图（设计 20261104-173254）：13 组细粒度部位 + 跨人距离边 + 跨人因果；G2 另加两项交互损失。
        ("G1", FLOW_BASE + ["--causal_cross", "--flow_kwargs", json.dumps({"local_k": 11, "part_scheme": "fine", "joint_graph": True}),
                            "--flow_recipe_weight", "1.0", "--flow_recipe_t_min", "0.5"]),
        ("G2", FLOW_BASE + ["--causal_cross", "--flow_kwargs", json.dumps({"local_k": 11, "part_scheme": "fine", "joint_graph": True}),
                            "--flow_recipe_weight", "1.0", "--flow_recipe_t_min", "0.5",
                            "--contact_curve_weight", "1.0", "--proximity_curve_weight", "1.0"]),
    ]
)
# 名称 -> (eval_ntu2p_v2 的 --flow_readout, 温度)；前三个是设计 3.3 的可采纳读出。
SCORE_READOUTS = OrderedDict(
    [
        ("ode0", ("sample", 0.0)),
        ("median16", ("median", 1.0)),
        ("median16t05", ("median", 0.5)),
        ("endpoint", ("endpoint", 1.0)),
    ]
)
ELIGIBLE = ("ode0", "median16", "median16t05")
# ReGenNet 上一轮已评价、且在本项目复核逐位一致的 J3 / FD2 结果：直接复用，不重复计算。
REGENNET_JITFLOW = Path("/home/rpartx3080/CodeSpace/ReGenNet/results/forecasting/ntu120_label/ntu2p_jitflow")
REUSE = {"FD2ref_endpoint.json": "FD2ref_endpoint.json", "J3_endpoint.json": "J3_endpoint.json",
         "J3_median.json": "J3_median16.json", "J3_samples.json": "J3_samples.json"}


def run_dir(opts, name, seed=None):
    seed = opts.seed if seed is None else seed
    if name == "J3":
        return SAVE / "ntu2p_jitflow_{}_J3-tmix_s{}_10000".format(opts.protocol, seed)
    return SAVE / "ntu2p_bandflow_{}_{}_s{}_{}".format(opts.protocol, name, seed, opts.steps)


def eval_readout(checkpoint, output, protocol, log, readout, temperature, split="val", export=None, extra=()):
    if not output.exists():
        command = [PYTHON, "eval/eval_ntu2p_v2.py", "--checkpoint", str(checkpoint), "--output", str(output),
                   "--baseline_checkpoint", protocol["base"], "--split", split, "--interaction_metrics",
                   "--flow_readout", readout, "--flow_temperature", str(temperature)] + common_args(protocol)
        if export:
            command += ["--export_arrays", str(export)]
        run(command + list(extra), log)
    return json.loads(output.read_text())


def train_candidate(opts, name, seed, protocol, log):
    directory = run_dir(opts, name, seed)
    checkpoint = directory / "ema/model{:09d}.pt".format(opts.steps)
    if not checkpoint.exists():
        say("训练 {} s{}".format(name, seed))
        run([PYTHON, "train/train_ntu2p_jitflow.py", "--save_dir", str(directory), "--num_steps", str(opts.steps), "--seed", str(seed),
             "--save_interval", str(opts.steps), "--baseline_checkpoint", protocol["base"]]
            + S2_5_RECIPE + FD2_ADDONS + CANDIDATES[name] + common_args(protocol), log)
    return checkpoint


def final_test_job(opts, seed, out):
    """原协议复训 + test 上只评一次 ode0；FD2 用同一命令重算 test（已评过的模型，不涉及选型）。"""
    protocol = PROTOCOLS[opts.protocol]
    log = out / "s{}.log".format(seed)
    checkpoint = train_candidate(opts, "BR", seed, protocol, log)
    fd2 = SAVE / "{}_{}_s{}_10000/ema/model000010000.pt".format(protocol["prefix"], FD2_CONFIG, seed)
    say("test 评价 s{}".format(seed))
    readout, temperature = SCORE_READOUTS["ode0"]
    results = OrderedDict()
    results["FD2"] = eval_readout(fd2, out / "FD2_s{}.json".format(seed), protocol, log, "endpoint", 1.0, split="test",
                                  export=out / "FD2_s{}_arrays.pt".format(seed))
    results["BR:ode0"] = eval_readout(checkpoint, out / "BR_ode0_s{}.json".format(seed), protocol, log, readout, temperature, split="test",
                                      export=out / "BR_ode0_s{}_arrays.pt".format(seed))
    return seed, results


def summarize_final(per_seed, out):
    rows = OrderedDict()
    for name in ("FD2", "BR:ode0"):
        values = [score_row(per_seed[seed][name]) for seed in sorted(per_seed)]
        keys = [key for key, value in values[0].items() if isinstance(value, (int, float))]
        rows[name] = OrderedDict(
            (key, OrderedDict([("mean", statistics.mean(v[key] for v in values)),
                               ("std", statistics.stdev(v[key] for v in values) if len(values) > 1 else 0.0),
                               ("per_seed", [v[key] for v in values])]))
            for key in keys)
    ref, new = rows["FD2"], rows["BR:ode0"]
    delta = OrderedDict((key, 100.0 * (new[key]["mean"] / ref[key]["mean"] - 1.0)) for key in ("xyz_mse", "xyz_mae", "mpjpe", "jerk_body"))
    delta["per_seed_mpjpe"] = [100.0 * (b / a - 1.0) for a, b in zip(ref["mpjpe"]["per_seed"], new["mpjpe"]["per_seed"])]
    (out / "summary.json").write_text(json.dumps(OrderedDict([("seeds", sorted(per_seed)), ("rows", rows), ("delta_pct", delta)]),
                                                 ensure_ascii=False, indent=2))
    lines = ["# BR:ode0 原协议 test（只评一次）", "", "3 seed 均值 ± 样本标准差；FD2 为同协议 3 seed 参照。", "",
             "|指标|FD2|BR:ode0|Δ%|", "|---|---:|---:|---:|"]
    for key in ref:
        change = "{:+.2f}".format(100.0 * (new[key]["mean"] / ref[key]["mean"] - 1.0)) if ref[key]["mean"] else "—"
        lines.append("|{}|{:.5f}±{:.5f}|{:.5f}±{:.5f}|{}|".format(key, ref[key]["mean"], ref[key]["std"], new[key]["mean"], new[key]["std"], change))
    lines.append("")
    lines.append("逐 seed Δmpjpe%：" + " / ".join("{:+.2f}".format(v) for v in delta["per_seed_mpjpe"]))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def job(opts, name, out):
    protocol = PROTOCOLS[opts.protocol]
    log = out / "{}.log".format(name)
    results = OrderedDict()
    if name == "FD2ref":
        checkpoint = SAVE / "{}_{}_s{}_10000/ema/model000010000.pt".format(protocol["prefix"], FD2_CONFIG, opts.seed)
        results["endpoint"] = eval_readout(checkpoint, out / "FD2ref_endpoint.json", protocol, log, "endpoint", 1.0)
        return name, results
    if name == "J3":
        checkpoint = run_dir(opts, name) / "ema/model000010000.pt"
    else:
        checkpoint = train_candidate(opts, name, opts.seed, protocol, log)
    say("评价 {}".format(name))
    for readout, (flow_readout, temperature) in SCORE_READOUTS.items():
        results[readout] = eval_readout(checkpoint, out / "{}_{}.json".format(name, readout), protocol, log, flow_readout, temperature)
    sample_path = out / "{}_samples.json".format(name)
    if not sample_path.exists():
        run([PYTHON, "eval/eval_ntu2p_jitflow_samples.py", "--checkpoint", str(checkpoint), "--output", str(sample_path)]
            + common_args(protocol)[:-2], log)
    results["sample"] = json.loads(sample_path.read_text())
    say("完成 {}".format(name))
    return name, results


def summarize(all_results, out):
    ref = score_row(all_results["FD2ref"]["endpoint"])
    rows = OrderedDict()
    for name, results in all_results.items():
        for readout, result in results.items():
            sample = readout == "sample"
            row = sample_row(result) if sample else score_row(result)
            row["delta_mpjpe_pct"] = 100.0 * (row["mpjpe"] / ref["mpjpe"] - 1.0)
            row["decision"] = decide(row, ref, sample)
            # 设计 3.3：只有多步采样读出可作为最终预测；endpoint 与单样本只作参照。
            row["decision"]["eligible"] = readout in ELIGIBLE
            row["decision"]["score_adopt"] = row["decision"]["score_adopt"] and readout in ELIGIBLE
            rows["{}:{}".format(name, readout)] = row
    (out / "summary.json").write_text(json.dumps(OrderedDict([("reference", "FD2ref:endpoint"), ("rows", rows)]), ensure_ascii=False, indent=2))
    header = ["行", "可采纳", "xyz_mse", "xyz_mae", "mpjpe", "Δmpjpe%", "dct_mid", "dct_high", "jerk", "穿地", "滑行", "步数",
              "相位f1-10", "excess_antic", "漏接触", "提分", "交互", "未过护栏"]
    lines = ["# 扩散完整预测候选 H-val 比较", "", "参照 FD2ref:endpoint（FD2 H-val EMA 10000 终点，同 seed）；判定按设计 4 节。", "",
             "|" + "|".join(header) + "|", "|" + "|".join(["---", "---"] + ["---:"] * (len(header) - 2)) + "|"]

    def fmt(value, digits=5):
        return "—" if value is None else ("{:." + str(digits) + "f}").format(value)

    for key, row in rows.items():
        decision = row["decision"]
        lines.append("|" + "|".join([
            key, "是" if decision["eligible"] else "参照", fmt(row["xyz_mse"]), fmt(row["xyz_mae"]), fmt(row["mpjpe"]),
            "{:+.2f}".format(row["delta_mpjpe_pct"]), fmt(row["dct_mid_ratio"], 3), fmt(row["dct_high_ratio"], 3), fmt(row["jerk_body"], 2),
            fmt(row["foot_penetration_ratio"], 4), fmt(row["slide_frame_ratio"], 3), fmt(row["step_count"], 3),
            fmt(row["gait_sep_corr_f01_10"], 3), fmt(row["primary_excess_antic"], 4), "{:.1f}".format(row["missed_contact_total"]),
            "是" if decision["score_adopt"] else "否", "是" if decision["interaction_adopt"] else "否",
            ",".join(decision["failed_guards"]) or "—"]) + "|")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="subjval")
    parser.add_argument("--candidates", nargs="+", default=["FD2ref", "J3"] + list(CANDIDATES))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--final_test", action="store_true", help="原协议复训 BR 并在 test 上只评一次 ode0（--seeds）")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    opts = parser.parse_args()
    if opts.final_test:
        if opts.protocol != "original":
            raise ValueError("--final_test 只用于 --protocol original（受试者留出 val 协议不评 test）")
        out = BASE / "ntu2p_jitflow" / opts.protocol / "bandflow_test"
        out.mkdir(parents=True, exist_ok=True)
        per_seed = OrderedDict()
        with ThreadPoolExecutor(max_workers=opts.workers) as pool:
            for future in as_completed([pool.submit(final_test_job, opts, seed, out) for seed in opts.seeds]):
                seed, value = future.result()
                per_seed[seed] = value
        summarize_final(per_seed, out)
        return
    out = BASE / "ntu2p_jitflow" / opts.protocol / "bandflow_s{}".format(opts.seed)
    out.mkdir(parents=True, exist_ok=True)
    source = REGENNET_JITFLOW / opts.protocol / "rep10000_s{}".format(opts.seed)
    for src, dst in REUSE.items():
        if (source / src).exists() and not (out / dst).exists():
            shutil.copyfile(source / src, out / dst)
    results = OrderedDict()
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futures = [pool.submit(job, opts, name, out) for name in opts.candidates]
        for future in as_completed(futures):
            name, value = future.result()
            results[name] = value
    summarize(OrderedDict((name, results[name]) for name in opts.candidates if name in results), out)


if __name__ == "__main__":
    main()
