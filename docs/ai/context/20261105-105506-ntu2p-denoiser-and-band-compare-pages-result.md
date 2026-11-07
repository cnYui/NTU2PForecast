<!-- prune:keep -->

# 去噪器内部详解页与 DCT 截断前后对比页：结果

计划：`20261105-105500-ntu2p-stg-denoiser-internals-page-plan.md`、`20261105-110300-ntu2p-band-limit-score-comparison-page-plan.md`。

## STG 去噪器内部

- 私有 Artifact：https://claude.ai/artifact/Cp8TvAxufEYSarxKAoquEw ；本地 `results/forecasting/ntu120_label/ntu2p_bandflow/denoiser_page/`（含 `ada_stats.json`）。
- 内容：两条路总览图；条件 c 的来历（`timestep_embedding` 64 维、×1000、32 个频率 → `time_mlp` 64→64→64 → + 动作 26×64、角色 2×64 → SiLU → [B,2,64]），t 滑块实时画正弦编码；token 的 6 项嵌入；一层内部 adaLN-Zero 图与公式（ada 64→1152 零初始化、RMSNorm 无仿射、4 头 × 48、qk RMSNorm、部位偏置 [4,2,5,5] 与时间偏置 [4,2,23] 六层共享、SwiGLU 512）；末层 adaLN 只有 shift/scale；参数表（BR seed0 实数，合计 4,570,427，6 个块 3,115,968，腿流 1,062,152）。
- 实算：训练后各层 \(\overline{|\alpha|}\) 随 t 的曲线（动作=推人）。第 2 层前馈门控 0.16（t=0）→ 0.36（t=1）；第 2–4 层注意力门控随 t 减小。时间 MLP 输出范数 t=0 约 9.3、t=1 约 17.7，动作/角色嵌入约 8。

## DCT 截断前后对比

- 私有 Artifact：https://claude.ai/artifact/XxjKfRtXBBNeukdRvcoKNU ；本地 `results/forecasting/ntu120_label/ntu2p_bandflow/band_compare_page/`（含 `band_scores.json`）。
- 两组只改阶数的对照（H-val seed0）：对照 A BR(11) vs BRfull(49)：mpjpe +0.18~+0.32%、xyz_mse −0.8~−2.5%、jerk +455%~+1657%；对照 B B(11) vs J3(20)：jerk +40~+95%，但 ode0 mpjpe −2.67%（单 seed，J3 seed1/2 收益明显小于 seed0；加配方分支后 BR 3 seed −2.24% 已超过 J3 seed1/2）。附 #010 单样本连续 6 帧条带。
- 两页发布前各截图看一次；去噪器页修正了总览图第 1 层文字与形状标注重叠、内部图说明文字压线。
