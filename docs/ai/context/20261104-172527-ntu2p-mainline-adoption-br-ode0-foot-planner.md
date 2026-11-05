<!-- prune:keep -->

# 主线采纳：扩散 BR:ode0 + 落脚规划层

## 决定

用户 2026-11-04 对"交替相关 −0.245（要求 ≤ −0.25）这一条，接受吗？"回答原文：**"接受，并入主线"**。

自此 NTU2P 主线为 **BR:ode0 + 落脚规划层**，取代 v2+FD2：

- 模型：`train/train_ntu2p_jitflow.py --denoiser stg --self_cond --t_uniform_mix 0.5 --flow_kwargs '{"local_k": 11}' --flow_recipe_weight 1.0 --flow_recipe_t_min 0.5` + FD2 损失配方（`scripts/run_ntu2p_bandflow.py` 中 `CANDIDATES["BR"]`），10000 step，EMA 0.999 终点。
- 读出：ode0（`--flow_readout sample --flow_temperature 0`，Heun 20 步）。
- 后处理：`--foot_planner`（`utils/ntu2p_foot_planner.py` 的 `DEFAULT_PARAMS`）。
- H-val 权重：`save/forecasting/ntu120_label/ntu2p_bandflow_subjval_BR_s{0,1,2}_10000/ema/model000010000.pt`；原协议权重 `ntu2p_bandflow_original_BR_s{0,1,2}_10000/ema/`。

## 采纳依据与被放宽的规则（如实记录）

| 依据 | 文档 |
|---|---|
| BR:ode0 H-val 3 seed：mpjpe −2.24%、mae −2.7%、jerk −23%；xyz_mse +2.8%，按用户授权的修订门限（≤ 1.05×）通过，原门限（≤ 1.005×）不通过 | `20261104-095145-ntu2p-bandflow-br-ode0-3seed-result.md`、`20261104-090141-ntu2p-bandflow-mse-tolerance-amendment.md` |
| 落脚规划层 H-val 3 seed：滑行帧比 .191 → .032（GT .034）、步数 1.37 → 1.69、穿地 .067 → .028、mpjpe +0.3%，抽帧目视通过；预登记的"左右交替相关 ≤ −0.25"未达（−0.245），用户接受 | `20261104-172148-ntu2p-foot-planner-result.md` |

两处都是在用户明确授权下放宽了预登记门限，不是事后自行调整。

## 已知局限（主线仍带着）

- 摆动能量（1.2–2 Hz）只有 GT 的约 26–29%，四肢摆得少。
- 规划层只管脚：模型选错的整体走向（例如 194 号窗 B 走进 A 的位置）与过宽站姿不受影响；步的时机由"脚被带开多远"决定，与 GT 步相位相关 .58。
- 交互时序（"手没碰到就躲"）未改善，primary_excess_antic .0308（FD2 .0247）。
- 随机样本概率质量未提升（单样本 ES ≈ .21，bootstrap .128）。

## test

原协议 3 seed 复训已在进行（计划 `20261104-163018-ntu2p-bandflow-br-ode0-original-protocol-test-plan.md`，按当时的主线只评 ode0）。主线改为带规划层后，**在同一批原协议权重上再对"ode0 + 落脚规划"在 test 上评一次**。两种读出的取舍已在 H-val 上决定，test 只报告、不再据此选择；两组 test 数字都如实给出。
