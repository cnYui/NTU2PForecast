<!-- prune:keep -->

# 主线架构流程与分数页面结果

- 计划：`20261113-103434-ntu2p-mainline-architecture-and-scores-page-plan.md`。
- 页面：私有 Artifact https://claude.ai/artifact/PWVE9K98X6AJeXiNWa7J3A ，本地副本 `results/forecasting/ntu120_label/ntu2p_bandflow/mainline_pgait_page/index.html`。

## 页面内容

1. **开头**：一句话说明任务；时间轴为观测 10 帧 + 预测 50 帧，20 fps；四个关键数字：
   - test mpjpe 0.1799，比 copy-last −39.2%；
   - 滑行帧比 0.035（GT 0.030）；
   - 步数 2.09（GT 2.19）；
   - 参数量：主体 4.57M + 步态模块 0.92M。
2. **架构流程图**（内联 SVG，两条泳道）：
   - **主体 BR**：规范化；扩散采样循环（Heun 20 步、39 次前向、z = 0 起步），每步依次是状态 z、120 个 token、STG × 6、部位头与最小二乘投影、腿流；之后 IDCT × ramp、刚体手、旋回相机系、骨架投影。
   - **步态模块 PGait**：PAE（虚线，只在训练时用）→ 相位预测器 F → 步态解码器 G → 观测锚定 → 从主体的髋做 FK。
   - 两部分组装后输出，不加后处理。
3. **训练分段**：主体 10000 step；PGait 依次为 PAE 4000、G 6000、F 12000 step，列出各段损失和增强。
4. **分数**：
   - 9 代主线的 test mpjpe 条形图（从 0 起画）；
   - 步态小多图：滑行、步数、左右交替、穿地，比较 FD2 / BR:ode0 / 当前，标出 GT 虚线；
   - 三个变化量：滑行 −82%、步数 +52%、mpjpe +0.46%；
   - 完整指标表，带 ±。
5. **代价**：mpjpe 未降、jerk 升高、手臂摆动偏小、交互无变化、训练与推理的骨盆来源不一致。

## 核对

- **token 数**：120 = 2 人 × 5 个部位 × 12 个 5 帧块，来自 `model/forecasting_ntu2p_jitflow.py` 的 `STG_PARTS`，以及 60 帧 ÷ patch 5。
- **PGait 参数量**：由 v4b 原协议 s0 checkpoint 实数得到，F 509,448、G 411,062、PAE 103,207。
- **历代 test mpjpe**：
  - ReGenNet `20260924-130524`：copy-last .29604、独立单人 .23974、s2_5 .19670；
  - `20260924-190149`：EMA .18816；
  - `20260925-181731`：rootdct .18455；
  - `20260926-005610`：v2 .18098；
  - 当前、BR:ode0、FD2 取自 `phase_gait_test/summary.json`。
- **发布前截图**：看了一次桌面浅色和窄屏深色。headless Chrome 的最小窗口宽度是 500 px；在 500 px 下测得页面没有横向溢出，架构图和表格都在各自的横向滚动容器里。
