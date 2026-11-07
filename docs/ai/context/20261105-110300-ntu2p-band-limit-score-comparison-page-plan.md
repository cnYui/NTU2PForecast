# DCT 带高频 vs 去高频的分数对比页计划

用户 2026-11-05："那之前 DCT 包含了高频的训练结果和现在不包含去掉了高频的这个频率的结果，做一个 HTML 展示一下两者的分数差异"。

## 用哪些结果（全部 H-val seed0、10000 step、EMA 终点，不新训练）

两组"只改局部 DCT 阶数"的干净对照：

- **对照 A（带高 t 配方分支）**：BR（root 8 / 局部 11）vs BRfull（49 / 49，不截断）。来源 `subjval/fullband_s0/summary.json`，读出 ode0、ode0 + 落脚规划、τ=1 单样本（同一命令评价）。
- **对照 B（不带配方分支）**：B（局部 11）vs J3（局部 20，上一轮的设置）。来源 `subjval/bandflow_s0/summary.json`，读出 ode0、median16、τ=1 单样本。
- 参照：FD2 端点（mpjpe .17223、jerk 20.40、dct_high .009）；GT 自身 jerk 306.7（含骨架拟合抖动）。

## 呈现

结论先行；jerk 对数柱状图（11 / 20 / 49 阶，各读出）；mpjpe 对比（标出约 0.8% 的单 seed 噪声）；两组完整指标表与 Δ%；#010 走开连续 6 帧单样本条带（BR vs BRfull）。如实写出对照 B 里 20 阶的 ode0 mpjpe 比 11 阶好 2.7%（单 seed，J3 的 seed1/2 收益明显小于 seed0），以及加了配方分支后这一差距消失。

本地 `results/forecasting/ntu120_label/ntu2p_bandflow/band_compare_page/`。
