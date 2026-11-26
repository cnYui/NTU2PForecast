# NTU2P FID 评估结果：MPJPE 与 FID 同时报告

用户 2026-11-26 要求（原话"写 FID 方案"→"开工，重训 1"）。起因是导师 11-25 建议同时报告 MPJPE 与 FID，以数值确认准确度与自然度的取舍，并要求所有比较方法都给出这两个数。设计与预登记见 `20261126-105120-ntu2p-fid-evaluation-design-and-plan.md`；识别网络与敏感性检查（在算任何方法的 FID 之前记录）见 `20261126-151545-ntu2p-fid-recognizer-and-sensitivity-result.md`。

## 结论

1. **3 与 4 的博弈由 FID 确认（预登记规则 2）**：4（BR:ode0 + PGait v4b）的 mpjpe 比 3（BR:ode0）高 0.46%，FID 低 16%（43.1 → 36.1）。3 个种子各自的配对 bootstrap 95% 置信区间都在 0 以下。另外两个识别网络下 FID 分别低 10% 和 13%，9 个"识别网络 × 种子"组合的置信区间全部 < 0，按规则 4 判为稳健。3 与 4 的主体相同，只有 6 个腿关节（双膝、双踝、双脚）不同，所以这个 FID 差异全部来自腿。
2. **2 与 3 之间也有类似的取舍**：3 的 mpjpe 比 2（FD2）低 0.8%，但 FID 更高。主识别网络上 3 个种子分别 +1.9（不显著）、+3.3、+8.1，另外两个识别网络上 6 个组合都显著。原因未分析。
3. **2 与 4 的 FID 分不出高低**：主识别网络上 4 低 2.6 分（2 个种子显著、1 个不显著）。另外两个识别网络上 2 反而更低（FD2 25.6 / 27.4，4 为 29.1 / 29.7）。FD2 的 FID 种子间波动最大（标准差 3.5）。不能说 4 的 FID 比 2 好。
4. **方法 1（ReGenNet 移植版）FID 252**，比独立单人 baseline（121）差得多，只比 copy-last（318）好。它是 DDIM50 随机采样；设计中担心"确定性读出在 FID 上吃亏"，但这里随机采样并没有带来 FID 优势。
5. **FID 对脚步只有弱敏感度**（见敏感性结果）：把真实动作的腿改成完全不迈步、只随骨盆滑动，全量 FID 只增加 2.8–6.8。抖动则完全看不出。所以汇报时脚步自然度仍需同时给出滑行帧比和步数。
6. 2/3/4 的绝对 FID（36–43）仍高于"换一批受试者的真实动作"（训练集 GT 对 test GT 为 23.1），与真实分布还有差距。

## 主表（原协议 test 1253 条，识别网络 s0）

2/3/4 为 3 个种子的均值 ± 标准差；FID 越低越好。

| 方法 | mpjpe | xyz_mse | FID | 识别 top1 | 滑行帧比 | 步数 | jerk |
|---|---:|---:|---:|---:|---:|---:|---:|
| copy-last | .2960 | .0690 | 317.8 | .172 | .008 | 0 | 5.6 |
| 独立单人 baseline | .2397 | .0466 | 121.3 | .471 | .587 | .03 | 7.4 |
| 1 ReGenNet 移植版（重训，1 seed，DDIM50） | .5797 | .1455 | 251.8 | .113 | .490 | .08 | 321.1 |
| 2 FD2 | .1805 ± .0004 | .02774 | 38.7 ± 3.5 | .884 | .203 | 1.46 | 20.7 |
| 3 BR:ode0 | **.1791** ± .0005 | .02888 | 43.1 ± 1.3 | .870 | .194 | 1.38 | 15.9 |
| 4 BR:ode0 + PGait v4b | .1799 ± .0005 | .02895 | **36.1** ± 1.0 | .868 | **.035** | **2.09** | 21.0 |
| GT | 0 | 0 | 0（训练集对 test：23.1） | .895 | .030 | 2.19 | 313 |

- mpjpe / 滑行帧比 / 步数 / jerk 由本次脚本从已导出数组复算，与此前文档一致（2 .18052、3 .17906、4 .17989）。
- GT 的 jerk 高是动捕抖动，此前文档已有说明。
- 单个 FID 的 bootstrap 置信区间因重采样含重复样本而明显偏上（例如 4@s0 为 37.4 [31.5, 53.6]），只用于参考；方法比较看下表的配对差值。

## 配对差值（识别网络 s0，同种子配对，1000 次 bootstrap）

| 配对 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| ΔFID(4 − 3) | −7.4 [−8.6, −6.3] | −6.1 [−7.1, −5.1] | −7.4 [−8.7, −6.3] |
| ΔFID(3 − 2) | +1.9 [−0.5, +4.1] | +3.3 [+1.1, +5.5] | +8.1 [+5.9, +10.4] |
| ΔFID(4 − 2) | −5.5 [−8.0, −3.3] | −2.8 [−5.2, −0.3] | +0.6 [−1.8, +3.1] |
| ΔFID(1 − 3) | +207 [+187, +228] | — | — |

## 识别网络稳健性（规则 4）

| 识别网络 | 2 FD2 | 3 BR:ode0 | 4 BR:ode0 + PGait | 4 相对 3 | ΔFID(4−3) 显著的种子数 | 单人 baseline | 1 移植版 | copy-last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s0（主） | 38.7 | 43.1 | 36.1 | −16.2% | 3/3 | 121.3 | 251.8 | 317.8 |
| s1 | 25.6 | 32.3 | 29.1 | −10.0% | 3/3 | 87.2 | 211.2 | 295.6 |
| s2 | 27.4 | 34.2 | 29.7 | −13.1% | 3/3 | 117.3 | 165.6 | 327.1 |

学习型方法之间的排序在三个识别网络上一致的只有：4 < 3、2 < 3，以及学习型 2/3/4 < 单人 baseline < 方法 1 < copy-last。2 与 4 的先后随识别网络变化。

## 方法 1 重训

- 8 月的权重已清理，按原 `args.json` 在 ReGenNet 仓库重训：seed 0、5000 step、batch 8。取 step 4000（当时在 val 上选中的步数，这次不重新选），用 DDIM50 在 test 上导出全部 1253 条。
- **重训版 test mpjpe .5797，原版 .5307**（+9%）。训练在 GPU 上不是逐位确定的，单 seed 的波动可能就有这么大。表中方法 1 的 FID 与 mpjpe 都来自重训版，两者口径一致；对外报 mpjpe 时须说明这一点。
- 产物（ReGenNet，已被 git 忽略）：权重 `save/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_fidretrain_s0_5000/`；指标与数组 `results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_fidretrain_s0_5000_test_step4000_ddim50/`（数组 800 MB，没有复制到本项目）。导出的窗口顺序、观测和 GT 与本项目导出逐条一致（GT 最大差 2e-6 m）。

## 复现

```bash
# 识别网络（各约 1 小时，GPU 共享时更慢）
python -m train.train_ntu2p_fid_recognizer --save_dir save/forecasting/ntu120_label/ntu2p_fid_recognizer_stgcn_s0 --seed 0 --device cuda:0
# 敏感性检查
python -m eval.eval_ntu2p_fid --stage sensitivity --recognizer save/forecasting/ntu120_label/ntu2p_fid_recognizer_stgcn_s0/recognizer.pt --output results/forecasting/ntu120_label/ntu2p_fid/sensitivity_rec_s0.json
# 各方法（--entry / --pair 的完整参数见 results/forecasting/ntu120_label/ntu2p_fid/methods_cli_args.txt）
python -m eval.eval_ntu2p_fid --stage methods --recognizer .../recognizer.pt --output results/forecasting/ntu120_label/ntu2p_fid/methods_rec_s0.json <entries> <pairs>
python scripts/plot_ntu2p_fid.py --input results/forecasting/ntu120_label/ntu2p_fid/methods_rec_s0.json --output results/forecasting/ntu120_label/ntu2p_fid/mpjpe_vs_fid_rec_s0.png
```

结果文件：`results/forecasting/ntu120_label/ntu2p_fid/{sensitivity,methods}_rec_s{0,1,2}.json`、`methods_rec_s{0,1,2}.md`、散点图 `mpjpe_vs_fid_rec_s0.png`。

## 新增代码

- `model/ntu2p_stgcn.py`：xyz STGCN（ReGenNet 结构）、29 关节子图、`recognizer_input`（规范化窗口）。
- `train/train_ntu2p_fid_recognizer.py`：训练识别网络，H-val 选 checkpoint，test GT 检查门槛，写 sha256。
- `eval/eval_ntu2p_fid.py`：敏感性检查与各方法 FID、配对 bootstrap、识别准确率、mpjpe/步态指标。Fréchet 距离用对称特征分解计算，与 ReGenNet 的 scipy 实现相差 1e-10。
- `scripts/plot_ntu2p_fid.py`：mpjpe–FID 散点图。

## 汇报

2026-11-26 已按用户确认，把主表数字、3 与 4 的博弈结论、2/4 分不出高低、FID 对脚滑弱敏感等要点连同散点图私信汇报给导师。
