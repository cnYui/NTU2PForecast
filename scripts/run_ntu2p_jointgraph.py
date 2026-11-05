"""关节级交互图 G1/G2：H-val seed0 训练、读出评价（均加落脚规划层）与预登记判定。

设计与预登记：docs/ai/context/20261104-173254-ntu2p-joint-interaction-graph-design-and-plan.md。
读出：ode0（主线读出，判定用）与 τ=0.3 单样本（只看四肢摆动）；参照为主线 BR 的同读出（seed0）。
"""

import argparse
import json
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from scripts.run_ntu2p_bandflow import BASE, PROTOCOLS, eval_readout, say, train_candidate

READOUTS = OrderedDict([("ode0plan", 0.0), ("t03plan", 0.3)])
PLANNER = ("--foot_planner",)
OVERLAP_NEAR, OVERLAP_GT_FAR = 0.10, 0.30


def body_min_distance(xyz):
    body = xyz[..., :22, :].double()
    return (body[:, :, 0].unsqueeze(3) - body[:, :, 1].unsqueeze(2)).norm(dim=-1).flatten(2).min(dim=-1).values


def wrong_overlap(arrays_path):
    """预测两人身体最近距离 < 0.10 m 而 GT ≥ 0.30 m 的帧占比（全部窗口与帧）。"""
    arrays = torch.load(arrays_path)
    pred, target = body_min_distance(arrays["methods"]["model"]), body_min_distance(arrays["target_xyz"])
    return float(((pred < OVERLAP_NEAR) & (target >= OVERLAP_GT_FAR)).double().mean())


def job(opts, name, out):
    protocol = PROTOCOLS["subjval"]
    log = out / "{}.log".format(name)
    if name == "BR":
        checkpoint = Path("save/forecasting/ntu120_label/ntu2p_bandflow_subjval_BR_s{}_10000/ema/model000010000.pt".format(opts.seed))
    else:
        checkpoint = train_candidate(opts, name, opts.seed, protocol, log)
    say("评价 {}".format(name))
    for readout, temperature in READOUTS.items():
        eval_readout(checkpoint, out / "{}_{}.json".format(name, readout), protocol, log, "sample", temperature,
                     export=out / "{}_{}_arrays.pt".format(name, readout), extra=PLANNER)
    say("完成 {}".format(name))
    return name


def row(out, name, readout):
    result = json.loads((out / "{}_{}.json".format(name, readout)).read_text())
    natural, timing, contact = result["naturalness"]["model"], result["interaction_timing"], result["contact"]["model"]
    values = OrderedDict((key, result["model_metrics"][key]) for key in ("mpjpe", "xyz_mse", "xyz_mae"))
    values.update(
        [
            ("primary_excess_antic", timing["primary_excess_antic"]["model"]),
            ("missed_contact_total", sum(timing["all_{:.2f}".format(t)]["model"]["missed_count"] for t in (0.20, 0.25, 0.30))),
            ("contact_rmse_push", contact["pushing"]["distance_rmse"]),
            ("contact_rmse_all", contact["all"]["distance_rmse"]),
            ("wrong_overlap", wrong_overlap(out / "{}_{}_arrays.pt".format(name, readout))),
            ("slide_frame_ratio", natural["slide_frame_ratio"]),
            ("step_count", natural["step_count"]),
            ("jerk_body", natural["jerk_body"]),
            ("foot_penetration_ratio", natural["foot_penetration_ratio"]),
            ("bone_rel_err_body", natural["bone_rel_err_body"]),
            ("dct_mid_ratio", result["articulation_gate"]["model_dct_mid_ratio"]),
        ]
    )
    return values


def decide(cand, ref):
    """设计第 6 节：交互三条 + 错误重叠 + 护栏。"""
    checks = OrderedDict(
        [
            ("接触前超额后退 ≤ 0.8×BR", cand["primary_excess_antic"] <= 0.8 * ref["primary_excess_antic"]),
            ("漏接触 ≤ BR", cand["missed_contact_total"] <= ref["missed_contact_total"]),
            ("推人接触距离 RMSE ≤ BR", cand["contact_rmse_push"] <= ref["contact_rmse_push"]),
            ("错误重叠帧比 ≤ BR", cand["wrong_overlap"] <= ref["wrong_overlap"]),
            ("护栏 mpjpe ≤ 1.01×BR", cand["mpjpe"] <= 1.01 * ref["mpjpe"]),
            ("护栏 xyz_mse ≤ 1.02×BR", cand["xyz_mse"] <= 1.02 * ref["xyz_mse"]),
            ("护栏 滑行 ≤ BR+0.01", cand["slide_frame_ratio"] <= ref["slide_frame_ratio"] + 0.01),
            ("护栏 步数 ≥ BR−0.1", cand["step_count"] >= ref["step_count"] - 0.1),
            ("护栏 jerk ≤ 24.5", cand["jerk_body"] <= 24.5),
            ("护栏 骨长 ≤ 1e-5", cand["bone_rel_err_body"] <= 1e-5),
            ("护栏 穿地 ≤ BR+0.01", cand["foot_penetration_ratio"] <= ref["foot_penetration_ratio"] + 0.01),
        ]
    )
    return checks


def swing_ok(t03, ode0):
    """τ=0.3 读出的摆动判定（次要）：dct_mid ≥ 0.40、mpjpe ≤ 1.01×同模型 ode0、jerk ≤ 24.5。"""
    return t03["dct_mid_ratio"] >= 0.40 and t03["mpjpe"] <= 1.01 * ode0["mpjpe"] and t03["jerk_body"] <= 24.5


def summarize(out, names):
    rows = OrderedDict(("{}:{}".format(n, r), row(out, n, r)) for n in names for r in READOUTS)
    ref = rows["BR:ode0plan"]
    result = OrderedDict([("rows", rows), ("decisions", OrderedDict()), ("swing", OrderedDict())])
    for name in names:
        result["swing"][name] = swing_ok(rows[name + ":t03plan"], rows[name + ":ode0plan"])
        if name != "BR":
            checks = decide(rows[name + ":ode0plan"], ref)
            result["decisions"][name] = OrderedDict([("checks", checks), ("pass", all(checks.values()))])
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    keys = list(next(iter(rows.values())))
    lines = ["# 关节级交互图 H-val", "", "|行|" + "|".join(keys) + "|", "|---|" + "---:|" * len(keys)]
    for key, values in rows.items():
        lines.append("|{}|{}|".format(key, "|".join("{:.4f}".format(v) for v in values.values())))
    for name, decision in result["decisions"].items():
        lines += ["", "{} 判定（ode0 + 规划层）：{}".format(name, "通过" if decision["pass"] else "不通过")]
        lines += ["- {}：{}".format(k, "过" if v else "未过") for k, v in decision["checks"].items()]
    lines += ["", "τ=0.3 摆动判定：" + "，".join("{} {}".format(n, "过" if ok else "未过") for n, ok in result["swing"].items())]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", default=["BR", "G1", "G2"])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    opts = parser.parse_args()
    opts.protocol = "subjval"
    out = BASE / "ntu2p_jitflow" / "subjval" / "jointgraph_s{}".format(opts.seed)
    out.mkdir(parents=True, exist_ok=True)
    # 主线 BR 的 ode0 + 规划层已评过（与本驱动同一命令）时直接复用；只有 seed0 导出了数组，其余 seed 由 job 重新评价。
    source = BASE / "ntu2p_jitflow" / "subjval" / "bandflow_s{}".format(opts.seed)
    if (source / "BR_ode0plan.json").exists() and (source / "BR_ode0plan_arrays.pt").exists():
        for name in ("BR_ode0plan.json", "BR_ode0plan_arrays.pt"):
            if not (out / name).exists():
                shutil.copyfile(source / name, out / name)
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        for future in as_completed([pool.submit(job, opts, name, out) for name in opts.candidates]):
            future.result()
    summarize(out, opts.candidates)


if __name__ == "__main__":
    main()
