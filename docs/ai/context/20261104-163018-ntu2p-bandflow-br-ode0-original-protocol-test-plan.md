# BR:ode0 原协议复训与一次 test：计划

关联：H-val 采纳 `20261104-095145-ntu2p-bandflow-br-ode0-3seed-result.md`；用户 2026-11-04 对"接下来对 BR:ode0 做原协议复训并评一次 test 吗？"回答"要"。

## 做法

- 配置与 H-val 完全相同（`scripts/run_ntu2p_bandflow.py` 中 `CANDIDATES["BR"]` + S2_5_RECIPE + FD2_ADDONS，10000 step，EMA 0.999），只换协议：原 manifest `ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json`、缓存 `ntu2p_xyz_seq_cache`（train 1758）、baseline `ntu2p_independent_single_person_o10_p50_cuda_retrain_s0_5000`。seed 0/1/2，最多两组并发。
- 读出固定为 **ode0**（τ=0、Heun 20 步、seed 2718；τ=0 时与采样 seed 无关），每个 seed 的 EMA 10000 终点只在 test（1253）上评价一次，带 `--interaction_metrics`。不评其它读出，不看原协议 val。
- 参照：FD2 原协议 3 seed EMA 10000 终点（已在 2026-09-26 评过 test，mpjpe .1805 ± .0005）。用同一命令在 test 上重算一次以得到交互指标；这不涉及选型，L2 须与存档 `eval_v2_test_000010000.json` 一致，否则停下排查。
- 报告：3 seed 均值 ± 样本标准差与逐 seed Δ；L2、摆动、自然度、步态、交互全部列出；生成 test 审查图（与 2026-09-26 FD2 test 审查相同的案例集，若可复用）。
- 无论结果如何都不再调整配置或重测 test；结果只陈述，不再"采纳/否决"——采纳已在 H-val 上按修订规则完成。

## 实现

`scripts/run_ntu2p_bandflow.py` 增加 `--final_test`：仅允许 `--protocol original`；训练 BR 后在 test 上评 ode0，FD2 重算 test，输出到 `results/forecasting/ntu120_label/ntu2p_jitflow/original/bandflow_test/`，并写 3 seed 汇总 `summary.{json,md}`。
