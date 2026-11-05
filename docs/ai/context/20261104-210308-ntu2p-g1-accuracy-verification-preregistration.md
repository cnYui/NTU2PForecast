# G1 精度副发现的验证：预登记

关联：`20261104-193546-ntu2p-joint-interaction-graph-result.md`（G1 在 H-val seed0 上 mpjpe −2.2%、xyz_mse −4.4%，未预登记的副发现）。用户 2026-11-04 选择"验证 G1 的精度提升"。

## 问题

G1（13 个关节组 + 跨人距离边 + 跨人因果，配置见 `scripts/run_ntu2p_bandflow.py` 的 `CANDIDATES["G1"]`）在精度上是否稳定优于主线 BR？读出与主线相同：ode0 + 落脚规划层。

## 做法

- 在 H-val 上补训 G1 seed1、seed2（10000 step，EMA 终点），配置与 seed0 完全相同；BR 的 seed1/2 已有。
- 以 `scripts/run_ntu2p_jointgraph.py --seed s` 评价（与 seed0 同一命令：`--flow_readout sample --flow_temperature 0 --foot_planner --interaction_metrics`）。
- **seed0 是发现这个提升的 seed，计入均值会偏乐观，判定以新的 seed1、seed2 为准**；3 seed 均值只作报告。

## 判定（在看到 seed1/2 任何结果之前固定）

参照：同 seed 的 BR:ode0 + 规划层（H-val：s1 mpjpe .16899 / mse .02012 / mae .08213；s2 .16781 / .01976 / .08156）。

**精度（全部须满足，用 seed1、seed2 的均值）**：
- mpjpe ≤ 0.992 × BR（改善 ≥ 0.8%，与历来提分门槛相同）；
- xyz_mse ≤ BR（不比主线差）；
- xyz_mae ≤ BR；
- 逐 seed：seed1、seed2 各自的 mpjpe 都低于同 seed 的 BR。

**护栏（seed1/2 均值）**：滑行帧比 ≤ BR + 0.01；步数 ≥ BR − 0.1；jerk ≤ 24.5（1.2 × FD2）；骨长 ≤ 1e-5；穿地 ≤ BR + 0.01；摆动 gate（dct_low ≥ 0.40、dct_mid ≥ 0.10）；接触前超额后退 ≤ 1.1 × BR；漏接触 ≤ 1.1 × BR。

**目视**：固定 9 例（与对比页相同）在 seed1 上抽帧，与 BR 相比不出现新的明显问题（腿拉长、膝反折、抖动、人物穿插加重）。

## 通过之后

G1 成为主线候选；须经用户确认后，在原协议上训练 3 seed 并只评一次 test（原协议 BR 的 test 已有，可直接对比）。不通过则如实报告，不再补 seed、不调门限。
