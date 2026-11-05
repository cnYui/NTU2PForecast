<!-- prune:keep -->

# G1 精度副发现的验证：结果

预登记：`20261104-210308-ntu2p-g1-accuracy-verification-preregistration.md`。副发现来源：`20261104-193546-ntu2p-joint-interaction-graph-result.md`（G1 在 H-val seed0 上 mpjpe −2.2%、xyz_mse −4.4%）。

## 决定

**按预登记规则不通过：G1 在 seed0 上的精度提升没有复现，判为 seed 噪声。** 精度四条全部未过，护栏全过。按预登记不再补 seed、不调门限；主线仍为 BR:ode0 + 落脚规划层，G1 不进原协议复训/test。

## 结果（H-val，10000 step EMA 终点，ode0 + 落脚规划层；汇总 `results/forecasting/ntu120_label/ntu2p_jitflow/subjval/g1_verification_summary.md`）

| seed | BR mpjpe | G1 mpjpe | Δ | BR xyz_mse | G1 xyz_mse | Δ | BR xyz_mae | G1 xyz_mae | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0（发现用，不计判定） | .16959 | .16580 | −2.23% | .02028 | .01937 | −4.49% | .08222 | .08055 | −2.02% |
| 1 | .16899 | .16895 | −0.02% | .02012 | .02039 | +1.36% | .08213 | .08213 | −0.01% |
| 2 | .16781 | .16987 | +1.23% | .01976 | .02064 | +4.45% | .08156 | .08251 | +1.17% |
| **seed1/2 均值（判定）** | .16840 | .16941 | **+0.60%** | .01994 | .02052 | **+2.89%** | .08185 | .08232 | **+0.58%** |
| 3 seed 均值 ± std（只报告） | .16879 ± .00091 | .16821 ± .00213 | −0.35% | .02005 ± .00026 | .02013 ± .00067 | +0.40% | .08197 ± .00036 | .08173 ± .00104 | −0.29% |

判定（seed1/2 均值）：mpjpe ≤ 0.992×BR 未过（+0.60%）；xyz_mse ≤ BR 未过；xyz_mae ≤ BR 未过；逐 seed 都低于 BR 未过（seed1 持平、seed2 +1.23%）。护栏全过：滑行 .035 vs .034、步数 1.64 vs 1.71、jerk 21.6、骨长 0、穿地、摆动 gate（dct_mid .31）、超额后退 .0282 vs .0297、漏接触 45.5 vs 45。

## 解读

1. seed0 的 −2.2% 是三个 seed 里最好的一次；G1 的 seed 间标准差（mpjpe .0021、xyz_mse .00067）是 BR 的 2.3–2.6 倍，3 seed 均值与 BR 相差 −0.35%，在噪声内。G1 的结构（13 组 token 312 个、距离边）让结果对初始化更敏感，而不是更准。
2. 这再次印证"单 seed 结果偏乐观"：J3:ode0 seed0 −2.12% 而 seed1/2 只有 −0.65/−1.02%；本次更极端，seed1/2 完全没有提升。以后的副发现都须另立预登记、以新 seed 判定。
3. 交互与步态上 G1 与 BR 基本相同（与 seed0 的交互判定结论一致），关节级交互图在本问题上没有带来可测量的收益。

## 目视

未渲染抽帧。预登记中目视只是通过所需的附加条件，精度四条已全部未过，目视结果不能改变判定，故跳过；seed0 上 G1 与 BR 的视频已在交互图结果文档中看过，两者几乎相同。

## 产物

- 判定脚本 `scripts/summarize_ntu2p_g1_verification.py`；`scripts/run_ntu2p_jointgraph.py` 新增 `--seed`（输出 `jointgraph_s{seed}/`）。
- 评价：`results/forecasting/ntu120_label/ntu2p_jitflow/subjval/jointgraph_s{1,2}/`（含 BR、G1 的 ode0plan / t03plan json 与导出数组）。各目录 `summary.md` 是上一份预登记的交互判定，seed1/2 上 G1 同样不过（超额后退、漏接触未过）。
- 权重：`save/forecasting/ntu120_label/ntu2p_bandflow_subjval_G1_s{1,2}_10000/ema/model000010000.pt`。两个 G1 并发训练，驱动在 G1 评价后汇总时因 BR 结果尚未生成而报 KeyError 退出，不影响 G1 的评价文件；随后的 BR 评价补齐了汇总。
