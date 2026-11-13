# 对照实验：不做 DCT 截断（≈ 直接在 xyz 时间序列上扩散）的预登记

用户 2026-11-05 问"直接在 xyz 时间序列上做扩散效果会不会不好"，确认"跑对照实验"。这是诊断实验，不是新主线候选：结果只回答"带限是否必要"，不改主线。

## 候选

**BRfull** = 主线 BR 的全部设置，只把 root 与局部的 DCT 阶数都从 8 / 11 改为 **49**（`--flow_kwargs {"root_k":49,"local_k":49}`）。未来 50 帧乘 ramp 后首帧为 0，可表示的子空间正好 49 维，所以 49 阶 = 不截断，等价于在"首帧残差为 0 的 xyz 残差序列"上扩散（基正交，只是坐标旋转）。保留 ramp 是为了只改"带限"这一个因素。系数尺度按训练脚本照常从数据估计。

预算：H-val seed0 × 10000 step，EMA 终点，与已有 BR seed0（`save/.../ntu2p_bandflow_subjval_BR_s0_10000`）同 seed 同预算，因此只新训一个模型。STG 训练逐运行不确定，单 seed 噪声约 0.8%，mpjpe 差异小于 0.8% 视为无差别。

## 评价（同一命令，对 BR 与 BRfull 都评）

- ode0（主线读出）、ode0 + 落脚规划（主线完整输出）、τ=1 单样本（看样本本身的抖动）。
- 指标：xyz_mse / xyz_mae / mpjpe，jerk，dct_high（> 2 Hz 能量相对 GT），dct_mid，滑行帧比，步数，穿地。

## 判定（看结果前固定）

- **"全频更差"**：任一成立 —— ode0 mpjpe ≥ 1.008 × BR；ode0 或单样本 jerk ≥ 1.5 × BR；单样本 dct_high ≥ 3 × BR。
- **"全频不差"**：ode0 mpjpe ≤ 1.008 × BR，且 ode0 与单样本 jerk 都 ≤ 1.2 × BR，且单样本 dct_high ≤ 1.5 × BR。
- 其余为"混合"，如实报告各项。
- 目视：#010 走开、#028 推人、#194 跟随三例，GT | BR | BRfull（ode0 + 落脚规划）抽帧；另看两者的单样本有无可见抖动。

## 实现

`scripts/run_ntu2p_fullband.py`：复用 `run_ntu2p_bandflow` 的训练与评价函数，输出 `results/forecasting/ntu120_label/ntu2p_jitflow/subjval/fullband_s0/`，不覆盖已有汇总。
