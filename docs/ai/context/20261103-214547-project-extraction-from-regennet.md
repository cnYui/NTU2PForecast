<!-- prune:keep -->

# 从 ReGenNet 提取 NTU2P 训练代码：过程与验证

## 背景

用户 2026-11-03 要求把 NTU 双人预测的训练代码单独提取为新项目，再在新项目里做"扩散网络替换 FD2 的 Mixer 主干与输出头，最终预测由扩散采样生成"。来源：`/home/rpartx3080/CodeSpace/ReGenNet` 工作区（分支 `feat/ntu2p-jit-flow-rl`，基于 `4323ac7`，含 2026-10-19 JiT-RL 轮未提交的改动）。

## 提取范围

- 用实际 import 追踪（在 regennet 环境下导入 `train/train_ntu2p_v2.py`、`eval/eval_ntu2p_v2.py`、`train/train_ntu2p_jitflow.py`、`train/train_ntu2p_jitflow_grpo.py`、`eval/eval_ntu2p_jitflow_samples.py` 与缓存构建、驱动、检查、分析、诊断脚本，收集仓库内被加载的模块），再静态检查函数内的延迟 import。共复制 53 个文件，路径与 ReGenNet 相同，逐字节 `cmp` 核对一致。
- 未提取：`model/forecasting_ntu2p_graphdiff.py`（接触图纠正器，已否决路线，延迟 import 会拖入 `diffusion/`、`model/cmdm.py`、`data_loaders/humanml/` 整套旧扩散代码）；`eval/action_xyz_classifier.py`（只在旧单人训练开启 action loss 时用）。`model/forecasting_ntu2p_v2.py` 的加载分派对 graphdiff checkpoint 改为显式报错，这是相对 ReGenNet 唯一的代码改动。
- `data_loaders/forecasting/__init__.py` 仍导入 InterHuman / NTU label 等数据集类（是 `ntu_2p_diffusion` 的 manifest 工具所在包），保留原样以免改动包结构。
- 数据：`dataset/`、`body_models/`、`ntu2p_xyz_seq_cache{,_subjval}` 为指向 ReGenNet 的软链接；两个 manifest、JiT 流 3 seed 汇总表复制。参照权重复制（约 240 MB）：FD2 H-val/原协议各 3 seed、J3/J4 H-val 3 seed 的 EMA 10000 终点与 args.json，独立单人 baseline 两份。
- 文档：复制 v2/FD2、受试者留出 val、Track B、接触图扩散、JiT-RL 的关键设计与结果文档（原文件名，不改内容），以及步数预算设计与架构 HTML 图。

## 验证

| 检查 | 结果 |
|---|---|
| FD2 H-val s0 EMA 终点，`eval_ntu2p_v2.py` 重新评估 vs ReGenNet 存档 `eval_v2_val_000010000.json` | 439 个数值项全部一致（mpjpe 0.172228） |
| J3 H-val s0 EMA 终点，endpoint 读出 + 交互指标 vs `rep10000_s0/J3_endpoint.json` | 778 个数值项全部一致 |
| FD2 训练 300 step（与原 run 同参数、同 seed）vs 原 `train_log.jsonl` | 300 步 × 12 项最大绝对差 0（逐位相同） |
| J3 训练 300 step vs 原日志 | 第 1–2 步相同，第 3 步起 1e-6 量级分歧后放大；同项目内连跑两次同样从第 3 步分歧（第 300 步 9.167 vs 8.938） |

结论：提取忠实。**新发现**：STG 去噪器（J3/J4）的训练在 GPU 上不是逐位确定的（FD2 是），因此 J 系列的"3 seed 标准差"里同时包含 seed 与运行非确定性；以后若需要严格复现，须单独排查（候选来源：注意力反向、`index_copy`/gather 的原子加），本轮不处理。验证产物：`results/verify/`（评估 json）；300 step 的临时权重已删除。

## 运行方式

所有命令在项目根目录、`PYTHONPATH=.`、用 `/home/rpartx3080/.local/micromamba/envs/regennet/bin/python` 运行；驱动脚本（`scripts/run_ntu2p_v2_screen.py`、`scripts/run_ntu2p_jitflow.py`）沿用 `sys.executable` 与相对路径，无需改动。
