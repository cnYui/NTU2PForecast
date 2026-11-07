# 动画架构图页面计划（主线扩散模型 + 原始 DCT 模型 FD2）

用户 2026-11-05："你完整的画一下模型的架构图，要有动画效果，输入是什么，输出是什么；还有原始的DCT的模型架构是什么"。

## 内容

1. **输入与输出**：用真实样本（H-val #028 推人、#010 走开）做骨架动画。前 10 帧是输入（观测，0.5 s），后 50 帧是输出（预测，2.5 s）；三栏同步播放 GT | FD2 | 主线（BR:ode0 + 落脚规划）。标注张量：输入 `obs_xyz [B,10,2,55,3]`（相机系、米、20 fps）+ 动作标签 `[B]`（26 类），输出 `[B,50,2,55,3]`。
2. **主线扩散模型完整架构**（纵向流程图，带动画）：规范化 → 采样循环（解码状态 z + 观测 → 120 个时空 token → STG Transformer ×6 → 部位输出头 → 最小二乘投回 DCT 得 x̂ → 腿流替换腿系数 → Heun 更新 z，20 步共 39 次前向）→ IDCT×ramp → 组装 55 关节 → 旋回相机系 + 观测末帧 → 骨架投影 → 落脚规划 → 输出。动画：数据包沿路径走过各级，进入循环后绕 20 圈、t 计数从 0 到 1；旁边的小画布同步播放"当前状态 z 解码出的 50 帧"，从 t=0 的静止（z=0 即 copy-last）逐步变成最终预测。z 的中间状态用 BR seed0 权重在该样本上真实计算（CPU，温度 0）。
3. **原始 DCT 模型 FD2 完整架构**（同样式）：规范化 → 观测位移取 30 阶 DCT 作 token（+ 末帧姿态、相对几何、动作/角色）→ InterMixer ×6（系数轴混合、FiLM 通道 MLP、人物混合）→ root 头（前 8 个 token）/ 局部头（前 20 个 token，含双手轴角）+ 腿流 → IDCT×ramp → 刚体手组装 → 旋回 + 末帧 → 骨架投影 → 输出。动画：数据包一次前向走完。
4. **两者对照**：哪些级相同、哪些被替换；参数量、前向次数、test 主要指标。

## 数字与结构来源

代码 `model/forecasting_ntu2p_intermixer.py`、`model/forecasting_ntu2p_jitflow.py`、`utils/ntu2p_foot_planner.py`；checkpoint 配置（FD2：hidden 256、6 块、input_dct_k 30、root 8、local 20、刚体手、腿流、骨架投影，3.65M；BR：STG 宽 192、4 头、6 块、patch 5、root 8、local 11、自条件、无跨人因果，4.57M）。test 数字取自 `20261104-182659-ntu2p-mainline-test-result.md`。

## 实现

- 导出脚本 `scripts/export_ntu2p_arch_demo.py`：从对比页数组取样本，转到场景规范系取 (X, Y) 侧视投影，身体 22 关节，毫米整数；另算 BR 采样过程中 6 个 t 的状态解码。
- 页面沿用对比页配色字体；SVG 流程图 + canvas 骨架；`prefers-reduced-motion` 时不自动播放，可单步。
- 本地 `results/forecasting/ntu120_label/ntu2p_bandflow/arch_page/`，发布为私有 Artifact；发布前截图看一次。
