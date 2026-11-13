# 主线架构流程与分数页面计划

用户 2026-11-13 原话："你用html画一下当前架构的流程，以及你提升的分数"。

## 内容

1. **架构流程图**：当前主线 BR:ode0 + PGait v4b，手绘内联 SVG，两条泳道。
   - **主体 BR**：
     - 规范化；
     - 扩散采样循环：Heun 20 步、39 次前向、从 z = 0 出发。每步依次是：状态 z（root DCT 8 阶 + 局部 DCT 11 阶）→ 120 个时空 token → STG Transformer × 6 → 部位头 → 最小二乘投回 DCT → 腿流；
     - IDCT × ramp → 刚体手组装 → 旋回相机系 → 骨架投影。
   - **步态模块 PGait**：
     - 相位预测器 F，以 BR 给出的未来骨盆速度为条件；
     - 步态解码器 G，输出 6 段骨方向和着地概率；
     - 从 BR 的髋做 FK，再做观测锚定；
     - 周期自编码器只在训练时使用，用虚线表示。
   - 标出交接处：BR 给 root、髋、上半身，PGait 给膝、踝、脚掌。注明不加后处理。
   - 另附训练分段：BR 10000 step；PGait 依次为 PAE 4000、G 6000、F 12000 step，列出各段损失。
2. **分数**：只用原协议 test（1253 条）、3 seed 均值，不含新评价。
   - **mpjpe 演进**：copy-last → 独立单人 → 残差精修 + 摆动恢复 → EMA → rootdct → v2 → FD2 → BR:ode0 → 当前。数字取自各代结果文档。
   - **步态指标小多图**：滑行帧比、步数、左右交替相关、穿地，比较 FD2 / BR:ode0 / 当前，标出 GT 参考线。
   - **完整表**：带标准差。
   - **代价照实写**：mpjpe 比 BR:ode0 高 0.46%，jerk 15.9 → 21.0，手臂摆动 dct_mid .24 偏小，交互指标不变。

## 数字来源

- 当前、BR:ode0、FD2：`results/forecasting/ntu120_label/ntu2p_jitflow/original/phase_gait_test/summary.json`。
- 历代：ReGenNet 的
  - `20260924-130524`：copy-last .29604、独立单人 .23974、s2_5 .19670；
  - `20260924-190149`：EMA .18816；
  - `20260925-181731`：rootdct .18455；
  - `20260926-005610`：v2 .18098。
- 参数量：BR 4.57M（架构页记录）；PGait 推理 0.92M（F 0.51M + G 0.41M），PAE 0.10M。由 checkpoint 实数得到。

## 实现

单页 HTML，内联 SVG，不用外部库，适配深浅两种主题。发布为私有 Artifact，本地副本放在 `results/forecasting/ntu120_label/ntu2p_bandflow/mainline_pgait_page/`。发布前截图看一次。
