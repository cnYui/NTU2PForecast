# NTU2P FID 评估设计与计划（MPJPE 与 FID 同时报告）

本文只是设计与计划，尚未实现、未训练、未评价。

## 起因

2026-11-25 导师的反馈（转述）：认可用户对 3 与 4 的解释，即 3（BR:ode0）的脚常拖着滑，4（BR:ode0 + PGait v4b）让腿真的迈步，但迈步时机对不准时 MPJPE 反而更大，准确度与自然度之间存在取舍；必要时可以牺牲一些自然度换取精度。导师建议同时报告 MPJPE 与 ReGenNet 也用过的自然度指标 FID，以数值确认这一取舍，并要求所有比较方法都给出这两个数。

用户 2026-11-26：写 FID 方案。

## 目标

1. 对汇报过的各比较方法（1 ReGenNet 移植版、2 FD2、3 BR:ode0、4 BR:ode0 + PGait v4b），外加 copy-last 与独立单人 baseline，在原协议 test（1253 条）上同时报告 MPJPE 与 FID。
2. 用数值回答"3 的 MPJPE 更低、4 更自然"是否也体现在 FID 上，即博弈关系能否用 FID 确认。
3. 事先确认 FID 能否看见脚滑。看不见时如实说明，自然度一侧改用滑行帧比和步数支撑。

## 现状

- **ReGenNet 的 FID**（`ReGenNet/eval/a2m/stgcn/evaluate.py`、`fid.py`）：用动作识别网络 STGCN 的倒数第二层特征（256 维，两人特征取均值）分别拟合真实动作与生成动作的高斯分布，算 Fréchet 距离。识别网络 `ReGenNet/recognition_training/ntu_smplx/checkpoint_0100.pth.tar` 在 NTU120 双人 xsub.train 上训练，输入为 SMPL-X rot6d（55 关节 + 1 个平移节点）、60 帧、glob + translation。ReGenNet 同时报告 FID_train、FID_test、识别准确率、Diversity 和 Multimodality。
- **我们的输出是 55 关节 xyz**，没有关节旋转，原识别网络不能直接用。
- **已导出的 test 预测数组**（`results/forecasting/ntu120_label/ntu2p_jitflow/original/`，每个都是 1253 × 50 × 2 × 55 × 3，含 `obs_xyz`、`target_xyz`、`actions`、`meta`；`methods.base` 是独立单人 baseline）：
  - 2 FD2：`bandflow_test/FD2_s{0,1,2}_arrays.pt`
  - 3 BR:ode0：`bandflow_test/BR_ode0_s{0,1,2}_arrays.pt`
  - 4 BR:ode0 + PGait v4b：`phase_gait_test/PG_ode0_s{0,1,2}_arrays.pt`
  - copy-last 由 `obs_xyz` 末帧直接得到。
- **方法 1 的权重已不存在**：`ReGenNet/save/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_cuda_full_s0_5000/` 只剩 `args.json` 和 `train_log.jsonl`，权重已在 2026-08-22 清理。test mpjpe .5307 的来源记录是 `ReGenNet/docs/ai/context/20260822-102600-ntu2p-diffusion-ldm-inter-full-training-result.md`。
- ReGenNet 仓库另有一个 xyz 时间卷积动作分类器（`eval/action_xyz_classifier.py`），2026-07 曾在旧协议上用于 FID；它不是 ReGenNet 的 FID 网络，本方案不用。

## 关键决策与取舍

### D1 特征网络：ReGenNet 同款 STGCN，输入换成 xyz，重新训练

把 ReGenNet 的 STGCN 结构原样移植过来（空间划分策略、边重要性加权、两人特征取均值），只把输入从 rot6d 改成关节 xyz（`in_channels=3`），并去掉平移节点（xyz 已含骨盆位置）。FID 公式直接复用 `calculate_frechet_distance`。

这样对导师可以这样说明：网络结构和 FID 公式都与 ReGenNet 一致，只有输入表示不同。

不采用的备选：把各方法的 xyz 用 IK 反解成 SMPL-X 旋转后喂原识别网络。

- 关节位置定不出骨骼自旋，只能由先验或优化补出，补出的误差会进入所有方法的 FID。
- 原网络在真实旋转上训练，IK 得到的 GT 本身就偏离它的训练分布，GT 也必须走同一套 IK 才公平，两层误差叠在一起，难以解释。
- 1253 × 60 帧 × 2 人 × 约 15 组预测的 IK 工程量大。

代价：两种做法得到的 FID 绝对值都不能与 ReGenNet 论文表格直接比较，只能在本协议内横向比较。

### D2 关节子集：身体 25 个关节加每只手 index1 / pinky1（共 29 个）

即 `train/train_ntu2p_v2.py` 中 `LOSS_JOINT_SUBSETS["hand2"]`。2、3、4 都用刚体手，设计上就不预测手指；GT 的 30 个手指关节有真实动作。保留全部手指时，FID 会被这部分与问题无关、各方法同样做不到的差异主导。图结构取 SMPL-X 运动树在这 29 个关节上的子图：index1 / pinky1 接到手腕，jaw / eyes 接到头。

### D3 输入窗口：观测 10 帧 GT + 预测 50 帧，共 60 帧，在场景规范系下

- 60 帧与 ReGenNet 识别网络的 `num_frames: 60` 相同，也就是 ReGenNet "整段动作"的口径。观测段对所有方法相同，各方法之间的差异来自预测段，以及第 10 → 11 帧的衔接。
- 用 `utils/ntu2p_canonical.py` 的 `canonical_frame(obs_xyz, ...)` 规范化，参数与主线模型相同（双人连线定 yaw，重合时回退相机 x 轴），只由观测决定。GT 与各方法用同一个变换，规范系的选择不偏向任何方法。识别网络训练时同样规范化。

### D4 识别网络：只用训练数据训练和选择，算任何方法的 FID 之前冻结

- 训练：受试者留出划分的 train（1709 条序列，`manifest_subjval_seed0.json` + `ntu2p_xyz_seq_cache_subjval`），用 `NTU2PXYZTrainSampler` 随机抽 60 帧窗口。超参数沿用 ReGenNet 的 `opt.yaml`：batch 64、lr 1e-4、Adam。
- 选择：每 500 step 在 H-val（247 条）上评估 top1，取最好的 checkpoint，最多 10000 step。
- 门槛（只看真实 test GT，与各方法无关）：沿用项目已有的 `GATE_THRESHOLDS`，即 top1 ≥ .80、top5 ≥ .90、balanced ≥ .55。不过门槛时先停下报告用户，**不在 test 上调识别网络**。
- 冻结：门槛通过后记录 checkpoint 的 sha256，之后才计算任何方法的 FID。
- 主结果用识别网络 seed 0。另训 seed 1 和 seed 2，只用来检查各方法的 FID 排序是否随识别网络的种子改变。

### D5 只报 FID_test，同时附带识别准确率；不报 FID_train、Diversity、Multimodality

- FID_test：test 上的预测对 test GT。
- FID_train 需要以训练集的观测为条件生成，而预测模型训练时见过这些条件，测到的是拟合程度，不是泛化。
- 2、3、4 都是确定性读出（ode0 / 回归），同一条件只有一个输出，Diversity / Multimodality 没有意义。
- 识别准确率（识别网络对预测窗口的 top1）同一次前向就能得到，作为辅助列报告。

### D6 统计口径

- 每个"方法 × 种子"算一个 FID，表中报 3 个种子的均值 ± 标准差。
- 方法间差异用配对 bootstrap：对 1253 个窗口的下标重采样 1000 次，GT 与两种方法取同一组下标，算 ΔFID 的 95% 置信区间；种子对应配对（s0 对 s0，依此类推）。
- 尺度参照：从训练集 GT 用固定种子抽 1253 个窗口，算它与 test GT 的 FID，代表"换一批受试者的真实动作"有多大差距。

### D7 敏感性检查（先做，用来判断 FID 能看见什么）

| 编号 | 构造 | 要回答的问题 |
|---|---|---|
| S0 | test GT 随机分两半（各约 626 条），两半互算 FID，重复 100 次 | 有限样本的噪声底 |
| S1 | test GT 预测段的 6 个腿关节（双膝、双踝、双脚）保持观测末帧相对骨盆的位置，随骨盆平移，其余关节不变 | 纯滑行、腿不动能否被看见 |
| S2 | test GT 预测段每关节每帧加 σ = 1 cm 的独立高斯噪声 | 抖动能否被看见（对应 BR 与 FD2 的 jerk 差异） |
| S3 | copy-last | 整体冻结（应明显变差，用作合理性检查） |

S1–S3 与 S0 用同样的二分方式：某一半的扰动版对另一半的 GT，样本量与 S0 相同，有限样本偏差也相同。

## 比较对象（原协议 test，1253 条）

| 方法 | 种子 | 数据来源 |
|---|---|---|
| copy-last | — | 由 `obs_xyz` 生成 |
| 独立单人 baseline | 1 | 已导出数组中的 `methods.base` |
| 1 ReGenNet 移植版（`L_dm + L_inter`，DDIM50） | 1 | **需重训**，见步骤 5 |
| 2 FD2 | 3 | 已导出 |
| 3 BR:ode0 | 3 | 已导出 |
| 4 BR:ode0 + PGait v4b | 3 | 已导出 |

每行同时列出 mpjpe、xyz_mse、FID（均值 ± 标准差）、识别 top1、滑行帧比、步数、jerk。另画一张 mpjpe–FID 散点图，标出误差棒。

## 预登记的解读规则（看到任何方法的 FID 之前确定）

1. **FID 能否看见某种缺陷**：S1 / S2 在 100 次二分上的 FID 中位数高于 S0 的 95 分位数，判为"能看见"。
2. **博弈是否由 FID 确认**（3 对 4）：
   - 已知 mpjpe(3) < mpjpe(4)。若 3 个种子 FID 均值 FID(4) < FID(3)，且配对 bootstrap 中 ΔFID(4 − 3) 的 95% 置信区间在至少 2 个种子上 < 0，则判为"FID 确认博弈"。
   - FID(4) < FID(3) 但置信区间含 0：判为"方向一致、不显著"。
   - FID(4) ≥ FID(3)：判为"FID 不支持"。此时自然度一侧以滑行帧比和步数为准，并结合 S1 的结论说明原因（FID 看不见脚滑，或 4 在别处变差）。
3. 其余方法对之间（1、2 对 3、4）只报告数值和置信区间，不另设判定。
4. 识别网络 seed 1 / 2 下 3 与 4 的 FID 大小关系与 seed 0 相反时，结论降级为"不稳健"。
5. FID 只用于评价：不用来选模型、选 checkpoint，也不加进训练损失；test 上不做任何选型。

## 解读时必须说明的局限

- **确定性读出在 FID 上天然吃亏**：点预测趋向条件均值，方差小于真实分布，FID 的协方差项会惩罚这一点。方法 1 是 DDIM50 单次采样（随机），可能出现 mpjpe 很差而 FID 不差的情况；4 重新引入迈步，腿部方差变大，这本身就可能拉低 FID。这一点要和"更自然"分开表述。
- FID 绝对值与 ReGenNet 论文不可比（输入表示、关节子集、窗口口径都不同）。
- N = 1253、特征 256 维时 FID 有正偏差，只能在同样本量下横向比较。
- 识别网络为动作分类训练，可能对脚部细节不敏感，S1 用来检验这一点。

## 实现（NTU2PForecast）

| 文件 | 内容 |
|---|---|
| `model/ntu2p_stgcn.py` | 从 ReGenNet `eval/a2m/recognition/models/stgcn.py` 及 `stgcnutils/`（tgcn、graph）移植 STGCN，新增 29 关节 xyz 布局；保留来源说明（MIT） |
| `train/train_ntu2p_fid_recognizer.py` | 训练识别网络：规范化窗口、H-val 选 checkpoint、test GT 门槛、写 sha256；`--device cuda:0`，不回退 CPU |
| `eval/eval_ntu2p_fid.py` | 读预测数组与 GT，提特征，算 FID、配对 bootstrap、识别准确率、S0–S3、训练集参照；输出 json、md 表和散点图 |

输出目录：识别网络 `save/forecasting/ntu120_label/ntu2p_fid_recognizer_stgcn_s{0,1,2}/`，结果 `results/forecasting/ntu120_label/ntu2p_fid/`。

方法 1 的重训在 ReGenNet 仓库完成：用原 `args.json` 运行 `train/train_ntu_2p_forecasting_diffusion.py`（seed 0，5000 step）；当时在 val 上选的是 step 4000，这次直接取 step 4000，不重新选。再用 `eval/eval_ntu_2p_forecasting_diffusion.py --split test --save_arrays` 以 DDIM50 导出全部 1253 条 xyz（需确认 `--save_array_limit` 设为 1253，且导出的是 FK 后的 xyz），复制到 `results/forecasting/ntu120_label/ntu2p_fid/method1/`。重训的 test mpjpe 与 .5307 一起报告；该训练可能不是逐位确定的。

## 步骤

1. 移植 STGCN，写训练和评价脚本；CPU 小样本自检：形状、两人交换后特征不变、GT 对 GT 的 FID ≈ 0。
2. 训练识别网络 seed 0 / 1 / 2，过门槛后冻结（GPU，单卡共享，启动前看 `nvidia-smi`；耗时以实测为准）。
3. 跑 S0–S3 和训练集参照，写入结果文档，再看任何方法的 FID。
4. 对 copy-last、独立单人 baseline、2、3、4 算 FID（只做前向，不训练）。
5. 方法 1 重训并导出（GPU，**需用户确认**）。
6. 汇总表、散点图、结果文档 `docs/ai/context/<时间戳>-ntu2p-fid-evaluation-result.md`，并拟给导师的回复草稿（发送前给用户看）。

## 待用户决定

- **是否重训方法 1**：导师要求所有比较方法都给出结果，不重训的话表里就缺第 1 行。
- **联合训练 A2 是否纳入**：A2 只在 H-val（247 条）上有结果。247 小于特征维数 256，协方差奇异，FID 不可靠。要纳入，需要在原协议上训练 A2 并跑一次 test，或者在 H-val 上每条序列多取几个窗口；两种做法都改变了此前"A2 不进 test"的决定。
- **S1 判定 FID 看不见脚滑时**，是否补一个只用下半身关节的 FID 作补充（需另训识别网络，并另写预登记）。
