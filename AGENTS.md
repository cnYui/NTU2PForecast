# NTU2PForecast AI 入口

> **只修改 `AGENTS.md`。** `CLAUDE.md` 是指向本文件的软链接（Claude Code 读 `CLAUDE.md`，Codex 等读 `AGENTS.md`）；不要编辑、删除或替换 `CLAUDE.md`。

## 项目定位

从 `/home/rpartx3080/CodeSpace/ReGenNet`（2026-11-03，分支 `feat/ntu2p-jit-flow-rl` 工作区）提取的 **NTU120 双人 xyz 动作预测**训练代码：观测 10 帧 → 预测未来 50 帧（20 fps），条件为 26 类动作标签，paired 双人 55 关节 xyz。只保留当前主线 v2+FD2 与 JiT 流的训练/评价链路；InterHuman、CMDM、旧 rot6d 扩散、接触图纠正器（graphdiff）等未提取，需要时回 ReGenNet 查。提取记录见 `docs/ai/context/20261103-214547-project-extraction-from-regennet.md`。

## 稳定约束

- 默认使用中文；文档、计划、总结和代码注释均使用中文，除非用户明确要求英文。
- 实现前必须先完成 design / plan；把上下文、决策、取舍和结果写入 `docs/ai/context/`，新增文件只建 `YYYYMMDD-HHMMSS-文件名.md`，不修改、不重命名历史文件。
- 代码注释写原因，不写过程；优先函数式、复用现有模块，保持 KISS/DRY。
- 训练必须显式 `--device cuda:0`，禁止 CUDA 不可用时静默回退 CPU。GPU 是与其它会话共享的单张 RTX 3080 10GB，启动前先看 `nvidia-smi` 上的外部进程。
- 等待后台进程用 PID 或完成标记，**不要用 `pgrep -f` 模式**（曾因此死锁 3.5 h）。
- 不声称 multi-person forecasting、interaction-aware、DCT+扩散、接触图或 RL 微调扩散首创。
- 代码托管：GitHub **公开**仓库 https://github.com/cnYui/NTU2PForecast （用户 2026-11-05 选择公开）。在 feature 分支提交后向 `main` 提 PR，没有自动合并 CI；提交前确认不含密钥与数据。上游 ReGenNet 的 MIT 许可证见 `LICENSE`。记录 `docs/ai/context/20261105-092449-github-public-repo-and-pr.md`。

## 环境与数据

- Python：`/home/rpartx3080/.local/micromamba/envs/regennet/bin/python`（系统 python3 没有 torch）。所有命令在项目根目录运行。
- `dataset/`、`body_models/` 与两个 xyz 序列缓存是指向 ReGenNet 的软链接，只读；不要在本项目里重建或改写它们。manifest 已复制：
  - 受试者留出 H-val（选型用）：`results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json` + 缓存 `ntu2p_xyz_seq_cache_subjval`（train 1709 / val 247）。
  - 原协议（采纳后复训 + 一次 test）：`results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json` + 缓存 `ntu2p_xyz_seq_cache`（train 1758，test 1253）。
- 参照权重已复制到 `save/forecasting/ntu120_label/`：FD2 H-val 3 seed（`ntu2p_v2subj_A6-F-A5f0.05-A4-GH0.5-FD2_s{0,1,2}_10000/ema/`）、FD2 原协议 3 seed（`ntu2p_v2_..._FD2_s{0,1,2}_10000/ema/`）、JiT 流 J3/J4 H-val 3 seed（`ntu2p_jitflow_subjval_J{3,4}-tmix_s{0,1,2}_10000/ema/`）、独立单人 baseline 两份（训练/评价入口需要 `--baseline_checkpoint`）。

## 实验协议

- 选型只看 H-val；采纳后在原协议上训练 3 seed、**只评估一次 test**，不在 test 上选型。
- 预算：筛选 5000 step（单 seed 噪声约 0.8%，结论须谨慎），最终数字 10000 step × 3 seed；`--ema_decay 0.999`，报告 `ema/` 终点，不在 val 上挑 checkpoint。
- 主指标 paired `xyz_mse / xyz_mae / mpjpe`，同时报告摆动（dct_low/mid/high）、自然度/步态（滑行帧比、步数、相位、穿地、jerk、骨长）、交互时序（`primary_excess_antic`、漏接触），并看审查图；新方案的采纳规则须在看结果前预登记。
- H-val 参照（FD2 EMA 10000 × 3 seed，`results/forecasting/ntu120_label/ntu2p_jitflow/subjval/rep10000_summary.md`）：mpjpe .17220 ± .00056，xyz_mse .01938，xyz_mae .08412，jerk 20.40，穿地 .0669，excess_antic .0247。原协议 test：FD2 mpjpe .1805 ± .0005。

## 当前研究入口

- **当前主线（用户 2026-11-04 "接受，并入主线"）：扩散 BR:ode0 + 落脚规划层**。STG 去噪器 + JiT 流（局部 DCT 11 阶 + 高 t 配方分支，`scripts/run_ntu2p_bandflow.py` 的 `CANDIDATES["BR"]`），读出 ode0（τ=0、Heun 20 步），评价加 `--foot_planner`。H-val 3 seed：mpjpe .1688、滑行帧比 .032、步数 1.69、jerk 21.8。采纳时用户授权放宽两条预登记门限（xyz_mse ≤ 1.05×；交替相关 −0.245 未达 −0.25）。采纳记录与已知局限见 `docs/ai/context/20261104-172527-ntu2p-mainline-adoption-br-ode0-foot-planner.md`。**原协议 test（3 seed）**：mpjpe .17971 ± .00057（FD2 .18052，仅 −0.45%）、xyz_mse +4.8%、滑行帧比 .203 → .030、步数 1.46 → 1.72、漏接触 +19%；精度收益在 test 上基本消失，确定的改进是腿，见 `docs/ai/context/20261104-182659-ntu2p-mainline-test-result.md`。新方案以它为对照，并须报告自然度/步态指标、抽帧目视。
- 上一代主线 v2+FD2（InterMixer 端到端 DCT 回归），入口 `train/train_ntu2p_v2.py`、`scripts/run_ntu2p_v2_screen.py`；H-val mpjpe .1722，原协议 test .1805 ± .0005；结果见 `docs/ai/context/20260926-165932-ntu2p-finger-dedup-result.md`、`20260926-162912-ntu2p-subject-holdout-val-result.md`。
- 上一轮 JiT 流（ReGenNet 2026-10-19，不采纳）：时空图 Transformer 去噪器的确定性端点 mpjpe −3.3% 但 jerk 1.26×；16 样本中位数/均值读出 mpjpe −1.09%/−0.73%（10000 × 3 seed），但 jerk 1.7–2.9×、dct_high 超护栏；单样本 +22%。见 `docs/ai/context/20261019-034335-ntu2p-jitflow-rl-final-result.md` 及其引用文档。
- 用户 2026-11-03 要求：**以扩散网络替换 FD2 的 Mixer 主干与输出头（③④），最终预测由扩散采样生成**，保留规范化、观测编码、DCT 表示、IDCT×ramp 解码、腿流、刚体手与骨架投影。只有多步采样读出（ode0 / median16 / median16t05）可作最终预测，endpoint 与单样本只作参照。设计 `docs/ai/context/20261103-215324-ntu2p-diffusion-full-prediction-design-and-plan.md`。
- 2026-11-03 第 1 阶段（seed0 × 10000）**负结果，未采纳**：局部 DCT 带限到 11 阶（≤2 Hz）使单样本 jerk 55 → 29；再加高 t 配方分支（BR）后，BR:ode0 mpjpe −1.81%、mae −2.6%、jerk 15.4（FD2 20.4）、护栏全过，**只有 xyz_mse +4.4%**（容许 +0.5%）。mse 增量的 86% 来自最差 10 个窗口（采样读出押错整体走向），去掉后只差 +0.68%。J3 新读出 3 seed 全部过不了 jerk；J3:ode0 seed0 −2.12% 而 seed1/2 −0.65/−1.02%，单 seed 结果偏乐观。是否接受 mse 取舍待用户决定，不能自行放宽。结果 `docs/ai/context/20261103-224829-ntu2p-diffusion-full-prediction-result.md`。
- 2026-11-04 用户授权（原话"接受，补 3 seed"）只对 BR:ode0 把 xyz_mse 门限放宽到 1.05×（`docs/ai/context/20261104-090141-ntu2p-bandflow-mse-tolerance-amendment.md`）。**BR:ode0 3 seed 均值在修订规则下通过，H-val 采纳**：mpjpe .16835（−2.24%，逐 seed −1.81/−2.42/−2.47）、mae −2.7%、mse +2.8%、jerk 15.8（FD2 20.4）；原规则下不通过。审查图显示步行以弓步 + 滑行完成、194 号窗两人重叠，1.2–2 Hz 摆动只有 GT 的 26%（FD2 64%），交互时序未改善。BR:endpoint 按原规则全过但不算扩散生成。**原协议复训与 test 尚未执行，须用户确认**。结果 `docs/ai/context/20261104-095145-ntu2p-bandflow-br-ode0-3seed-result.md`；权重 `save/forecasting/ntu120_label/ntu2p_bandflow_subjval_BR_s{0,1,2}_10000/ema/`。
- 2026-11-04 用户要求"腿真正迈步、不漂移，视频效果优先，评价要自己抽帧看"。诊断：GT 经 11 阶 DCT 编解码仍正常迈步，表示不是瓶颈；模型把步相位平均掉，脚随骨盆平移。新增**落脚规划层**（`utils/ntu2p_foot_planner.py`，`eval/eval_ntu2p_v2.py --foot_planner`，不训练，只改双腿膝/踝/脚掌，参数全部来自 train GT）。v1（摆动固定 21 帧，来自带抖动的接触判定）抽帧可见一字马弓步，不通过；v2 摆动按步长自适应（5–12 帧）。BR:ode0 + v2 在 H-val 3 seed：滑行帧比 .191 → .032（GT .034）、步数 1.37 → 1.69、穿地 .067 → .028、mpjpe +0.3%，抽帧目视通过；预登记的"左右交替相关 ≤ −0.25"差一点（−0.245），是否接受待用户决定，参数不再调。结果 `docs/ai/context/20261104-172148-ntu2p-foot-planner-result.md`。规划层不改骨盆走向，194/216 号窗的走错方向、过宽站姿仍在。对比页 `results/forecasting/ntu120_label/ntu2p_bandflow/compare_page/index.html`。
- 2026-11-04 关节级交互图（用户要求"不同关节之间的图神经网络"，让"A 的手碰到 B 的肩之后 B 才后退"）**负结果**：G1（13 个关节组 + 跨人距离边 + 跨人因果）与 G2（再加手—躯干距离曲线、两人最近距离曲线损失）在 H-val seed0 都不过预登记交互判定；推人接触—后退曲线与抽帧显示三者反应时机几乎相同，194 号窗两人重叠仍在。副发现（seed0 上 G1 mpjpe −2.2%、xyz_mse −4.4%）经预登记补 seed1/2 验证**未复现**：seed1/2 均值 mpjpe +0.60%、xyz_mse +2.9%，精度四条全不过，判为 seed 噪声（G1 的 seed 间标准差是 BR 的 2.3–2.6 倍）；不再补 seed，G1 不进 test。结果 `docs/ai/context/20261104-193546-ntu2p-joint-interaction-graph-result.md`、验证 `docs/ai/context/20261104-225520-ntu2p-g1-accuracy-verification-result.md`；设计 `docs/ai/context/20261104-173254-ntu2p-joint-interaction-graph-design-and-plan.md`。单 seed 副发现一律另立预登记、以新 seed 判定。
- 2026-11-02 ~ 11-04 会话的完整上下文（用户原话、决定、数字、产物、待办）：`docs/ai/context/20261104-173814-session-context-20261102-20261104.md`。
- STG 去噪器（J3/J4/B/BR）的训练在 GPU 上不是逐位确定的（FD2 是），见提取记录。
