# F13（只细分部位）补 seed 验证：预登记

用户 2026-11-06："补 seed 确认"。F13 在 seed0 上 mpjpe −0.92%（刚过 −0.8% 门槛），漏接触与左右交替相关退步（`20261106-100630-ntu2p-fine-parts-and-skeleton-graph-result.md`）。G1 曾在 seed0 上 −2.2%、补 seed 后消失，所以判定**只用新补的 seed1、seed2**，seed0 与 3 seed 均值只作报告。

## 做法

F13 配置不变（`part_scheme="fine"`，无距离边、无跨人因果），H-val seed1、seed2 各 10000 step、EMA 终点，并行训练；读出 ode0 + 落脚规划，与同 seed 的 BR 配对（`subjval/jointgraph_s{1,2}/BR_ode0plan.json`：s1 mpjpe .16899 / mse .02012 / mae .08213；s2 .16781 / .01976 / .08156）。

## 判定（在看到 seed1/2 任何结果之前固定）

**精度（全部须满足，seed1/2 均值）**：mpjpe ≤ 0.992 × BR；xyz_mse ≤ BR；xyz_mae ≤ BR；逐 seed：seed1、seed2 的 mpjpe 都低于同 seed BR。

**护栏（seed1/2 均值）**：滑行 ≤ BR + 0.01；步数 ≥ BR − 0.1；jerk ≤ 24.5；骨长 ≤ 1e-5；穿地 ≤ BR + 0.01；摆动 gate；接触前超额后退 ≤ 1.1 × BR；漏接触 ≤ 1.1 × BR；**左右交替相关 ≥ BR − 0.05**（seed0 上下降 0.18，专门设这一条）。

**目视**：固定 4 例（#028、#002、#194、#010）在 seed1 上抽帧，与 BR 相比不出现新的明显问题。

## 之后

通过：F13 成为主线候选，须用户确认后在原协议上训练 3 seed、只评一次 test。不通过：如实报告，不再补 seed、不调门限。实现：`scripts/run_ntu2p_partgraph.py --seed s --names F13`，判定脚本 `scripts/summarize_ntu2p_f13_verification.py`。
