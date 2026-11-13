<!-- prune:keep -->

# 主线采纳：BR:ode0 + 相位流形步态模块 PGait v4b

用户 2026-11-07 原话："并入主线"。

## 新主线

**BR:ode0 + PGait v4b**。全部组件都参与训练，评价时不加任何后处理。

- **主体**：扩散 BR，即 STG 去噪器 + JiT 流，局部 DCT 11 阶 + 高 t 配方分支；读出 ode0（τ = 0、Heun 20 步）。主体给出 root、髋和上半身。
- **腿**：PGait v4b 生成膝、踝、脚掌，由三部分组成：
  - 周期自编码器：只在训练时提供 GT 相位；
  - 步态解码器 G：GT 相位训练，感知地面，以观测腿姿为条件，按 FK 保持骨长；
  - 相位预测器 F：圆周相位、量值、自洽支撑、观测重建，以及经冻结 G 回传的腿部 L2（权重 3）。

  输出经观测锚定（τ = 4 帧）。

**两段分开训练**：主体按 `scripts/run_ntu2p_bandflow.py` 的 `CANDIDATES["BR"]` 训练；步态模块按 `train/train_ntu2p_phase_gait.py` 训练，参数为 `--ground_aware --penetration_weight 10 --obs_recon_weight 1.0 --f_steps 12000 --obs_pose_input --anchor_tau 4 --vel_lowpass_order 8 --leg_l2_weight 3.0`，每个 seed 约 10 分钟，只用 train split。

**评价**：`eval/eval_ntu2p_v2.py --checkpoint <BR ema> --flow_readout sample --flow_temperature 0 --gait_head <phase_gait.pt>`。

## 主数字

| | mpjpe | xyz_mse | xyz_mae | 滑行帧比 | 步数 | 左右交替相关 | 穿地 | jerk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H-val 3 seed | .16894 | .01994 | .08208 | .029 | 2.10 | −.31 | .023 | 20.8 |
| 原协议 test 3 seed | **.17989 ± .0005** | .0290 | .0869 | .035 | 2.09 | −.30 | .011 | 21.0 |

- 相对上一主线 BR:ode0 不加后处理：test 上 mpjpe +0.46%，滑行 .194 → .035，步数 1.38 → 2.09。
- 相对 FD2：mpjpe −0.35%。

## 权重

- H-val：`save/forecasting/ntu120_label/ntu2p_bandflow_subjval_BR_s{0,1,2}_10000/ema/model000010000.pt`，加 `save/forecasting/ntu120_label/ntu2p_phase_gait_v4b_subjval_s{0,1,2}/phase_gait.pt`。
- 原协议：`save/forecasting/ntu120_label/ntu2p_bandflow_original_BR_s{0,1,2}_10000/ema/model000010000.pt`，加 `save/forecasting/ntu120_label/ntu2p_phase_gait_v4b_original_s{0,1,2}/phase_gait.pt`。
- 主体与步态模块同 seed 配对。

## 已知局限

- mpjpe 比不加步态模块高约 0.4–0.5%：未来时机不确定时，迈步本身会增加 L2。
- 步态模块用 GT root 训练，推理时接 BR root。
- 走错方向（#194、test #010）、被推后不后仰、手没碰到就躲，都属于主体问题，本模块不涉及。
- 手臂摆动仍偏小。

## 后续

新方案以本主线为对照，并须同时报告自然度/步态指标、抽帧目视。候选方向：

- 主体与步态端到端联合训练，目标是降低 mpjpe；
- 把相位流形扩展到全身（手臂）；
- 主体的走向/意图预测。

结果与过程见：
- `20261107-112431-ntu2p-phase-gait-v4b-original-protocol-test-result.md`
- `20261107-105943-ntu2p-phase-gait-v4-3seed-result.md`
- 设计 `20261107-095645-ntu2p-phase-manifold-gait-design-and-preregistration.md`
