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

- **主线仍为 v2+FD2**（InterMixer 端到端 DCT-Mixer + 脚接触 + 骨架投影 + 腿部专用流 + 手指损失去重），入口 `train/train_ntu2p_v2.py`、`eval/eval_ntu2p_v2.py`、`scripts/run_ntu2p_v2_screen.py`；结果见 `docs/ai/context/20260926-165932-ntu2p-finger-dedup-result.md`、`20260926-162912-ntu2p-subject-holdout-val-result.md`。
- 上一轮 JiT 流（ReGenNet 2026-10-19，不采纳）：时空图 Transformer 去噪器的确定性端点 mpjpe −3.3% 但 jerk 1.26×；16 样本中位数/均值读出 mpjpe −1.09%/−0.73%（10000 × 3 seed），但 jerk 1.7–2.9×、dct_high 超护栏；单样本 +22%。见 `docs/ai/context/20261019-034335-ntu2p-jitflow-rl-final-result.md` 及其引用文档。
- 用户 2026-11-03 要求：**以扩散网络替换 FD2 的 Mixer 主干与输出头（③④），最终预测由扩散采样生成**，保留规范化、观测编码、DCT 表示、IDCT×ramp 解码、腿流、刚体手与骨架投影。设计与预登记见本目录下当天的 design-and-plan 文档。
