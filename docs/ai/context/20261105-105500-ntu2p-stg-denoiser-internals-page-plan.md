# STG 去噪器内部详解页计划

用户 2026-11-05："里面具体的模型架构你可以用一个html来展开详细讲解一下，比如这些都是怎么加进来的：噪声时刻 t / 正弦编码 → MLP / + 条件 → adaLN"。

## 内容（全部对照 `model/forecasting_ntu2p_jitflow.py` 与 BR seed0 权重）

1. 总览图：token 一路（patch 嵌入 + 各种位置嵌入 → 6 个块 → 末层 adaLN → 部位输出头 → 投回 DCT → 腿流）与条件一路（t → 正弦编码 → MLP，+ 动作嵌入 + 角色嵌入 → SiLU → c）并排，c 接到每一层。
2. 条件一路：`timestep_embedding(t, 64)`（32 个频率 \(\omega_i=10000^{-i/32}\)，输入 1000t，cos/sin 各 32 维）→ `time_mlp` 64→64→64 → 加 `action_embed[a]`（26×64）与 `role_embed[p]`（2×64）→ SiLU → c ∈ [B,2,64]。交互：拖 t 看 64 维编码（按公式在页面里实算）。
3. 一个块内部：adaLN-Zero（`ada`: 64 → 6×192，零初始化）给出 shift/scale/gate；RMSNorm（无仿射）→ 调制 → 注意力（4 头 × 48，qk RMSNorm，部位对偏置 [4,2,5,5]、相对时间偏置 [4,2,23]，6 层共享）→ ×gate 残差；SwiGLU（192→512→192）同样调制。真实数据图：训练后各层 |gate| 随 t 的变化（动作=推人，角色 A/B）。
4. token 嵌入各项：部位 patch 瓶颈（in→128→192）、part_pos [5,192]、patch_pos [12,192]、obs_flag [2,192]、role_pos [2,192]、static_proj（几何 6 + 末帧姿态 165 → 192）。
5. 输出：末层 adaLN（只有 shift/scale）、部位输出头（零初始化）、最小二乘投回 DCT、腿流（也读 c）。
6. 参数表（BR seed0 实数，合计 4,570,427）。

本地 `results/forecasting/ntu120_label/ntu2p_bandflow/denoiser_page/`；公式用 cdnjs MathJax 3.2.2；发布前截图看一次。
