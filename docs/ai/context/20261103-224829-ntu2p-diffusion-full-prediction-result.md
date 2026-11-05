# NTU2P 扩散完整预测（带限流 + 采样读出）：第 1 阶段结果

设计与预登记：`20261103-215324-ntu2p-diffusion-full-prediction-design-and-plan.md`。全部为受试者留出 H-val（train 1709 / val 247），未读取 test。

## 决定

**按预登记规则，没有候选通过提分规则，本轮记为负结果；不补 B/BR 的 seed1/2，不延长训练，不调宽门限。主线仍为 v2+FD2。**

最接近的是 **BR:ode0**（局部 DCT 11 阶 + 高 t 配方分支，τ=0 确定性 ODE 采样）：mpjpe −1.81%、xyz_mae −2.6%、jerk 15.4（FD2 20.4），全部护栏通过，**只有 xyz_mse +4.4%**（容许 +0.5%）。是否在 mse 上接受这一取舍、以 3 seed 复现，须由用户决定。

## 第 0a 阶段：带宽选择（只读 train）

`scripts/diag_ntu2p_band_oracle.py` → `results/forecasting/ntu120_label/ntu2p_bandflow/oracle_train.json`。GT 经同一编解码（含骨架投影）的 oracle：

| 局部 K | oracle xyz_mse | oracle mpjpe | Δmse vs K20 |
|---:|---:|---:|---:|
| 10 | .000689 | .0220 | +1.94e-4 |
| **11** | .000653 | .0212 | **+1.58e-4** |
| 12 | .000625 | .0205 | +1.30e-4 |
| 14 | .000580 | .0194 | +8.5e-5 |
| 20 | .000495 | .0172 | 0 |

Δmse(K11) = FD2 H-val mse 的 0.82%，低于预登记的 2% 回退线，按规则取 K=11。GT 局部姿态在评价口径三频带上的能量（每窗）为 low 100.7 / mid 6.46 / high 6.76：>2 Hz 的能量与 1.2–2 Hz 步态频段相当，与"高频带多为拟合抖动"一致。

## 第 1 阶段：seed0 × 10000 step（EMA 终点）

完整表：`results/forecasting/ntu120_label/ntu2p_jitflow/subjval/bandflow_s0/summary.md`。参照为 FD2 H-val s0 EMA 10000。

| 行 | 可采纳 | xyz_mse | xyz_mae | mpjpe | Δmpjpe | dct_mid | dct_high | jerk | 穿地 | 未过 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FD2 参照 | — | .01930 | .08420 | .17223 | — | .626 | .009 | 20.40 | .0640 | — |
| J3:ode0 | 是 | .01974 | .08204 | .16857 | −2.12% | .269 | .040 | 33.70 | .0552 | mse, jerk, 相位 |
| J3:median16t05 | 是 | .01958 | .08193 | .16829 | −2.29% | .298 | .049 | 42.03 | .0562 | mse, jerk |
| B:ode0 | 是 | .02129 | .08412 | .17319 | +0.56% | .110 | .007 | 17.25 | .0611 | mpjpe, mse |
| B:median16t05 | 是 | .02082 | .08346 | .17172 | −0.30% | .121 | .012 | 27.18 | .0631 | 提分不足, mse, jerk |
| **BR:ode0** | 是 | .02015 | .08204 | .16911 | **−1.81%** | .232 | .011 | **15.37** | .0680 | **只差 mse（+4.4%）** |
| BR:median16 | 是 | .01936 | .08207 | .16885 | −1.96% | .384 | .069 | 46.34 | .0611 | jerk, dct_high |
| BR:median16t05 | 是 | .01967 | .08146 | .16777 | −2.59% | .260 | .023 | 27.09 | .0622 | mse（+1.9%）, jerk（1.33×） |
| B:endpoint | 参照 | .01976 | .08293 | .17026 | −1.14% | .597 | .011 | 18.29 | .0547 | （不可采纳） |
| BR:endpoint | 参照 | .01943 | .08258 | .16950 | −1.58% | .588 | .012 | 18.66 | .0540 | （不可采纳） |

单样本（τ=1，16 个平均）与 Energy Score：

| 候选 | 单样本 mpjpe | jerk | dct_high | dct_mid | 穿地 | ES |
|---|---:|---:|---:|---:|---:|---:|
| J3 | .2099 | 55.4 | .196 | 1.12 | .105 | .204 |
| B | .2120 | 28.8 | .036 | 0.47 | .109 | .207 |
| BR | .2142 | 33.4 | .134 | 2.25 | .112 | .208 |

## J3 新读出 3 seed（第 0b 阶段）

`results/forecasting/ntu120_label/ntu2p_jitflow/subjval/bandflow_J3_3seed_summary.md`：

| 读出 | mpjpe Δ（逐 seed） | xyz_mse | jerk | 判定 |
|---|---|---:|---:|---|
| ode0 | −1.26%（−2.12/−0.65/−1.02） | .02020（+4.2%） | 34.86（1.71×） | mse, jerk |
| median16t05 | −1.72%（−2.29/−1.28/−1.61） | .01992（+2.8%） | 43.10（2.11×） | mse, jerk, dct_high |
| median16（上一轮） | −1.09% | .01980（+2.2%） | 59.90 | mse, jerk, dct_high |

**seed0 偏乐观**：J3:ode0 在 seed0 上 −2.12%，另两个 seed 只有 −0.65/−1.02%。BR:ode0 只有 seed0，其 −1.81% 应按同样的不确定性理解。

## 归因与诊断

1. **带限降低抖动（主要目的达成）**：B 相对 J3，单样本 jerk 55 → 29、dct_high .196 → .036；BR:ode0 的 jerk 15.4 低于 FD2。
2. **带限单独会损失精度，采样路径配方把它补回**：同为 K=11，BR:ode0 比 B:ode0 的 mpjpe 好 2.4 个百分点，BR:endpoint 也比 B:endpoint 好 0.4 个百分点。配方分支让交付的末步 x̂ 受骨架、脚、步态与幅度项约束。
3. **xyz_mse 失败来自"押错模式"的少数大误差**（`results/forecasting/ntu120_label/ntu2p_bandflow/diag/BR_ode0_s0.json` 与导出数组；逐窗对比 FD2）：BR:ode0 在 153/247 个窗口上 mpjpe 更低，逐窗 mpjpe 中位数 .1554 vs .1598；但 Δmse 的 86% 来自最差的 10 个窗口（动作分散：19×3、21×2 等），去掉这 10 个窗口两者 mse 只差 +0.68%。分 horizon：0–0.5 s mse 更低（.00423 vs .00467），0.5 s 之后更高；root mse +5.6%、局部 +2.4%。确定性回归对多模态未来给出条件均值，在 mse 上天然占优；单条采样读出在少数窗口选错整体走向时，平方误差被放大。
4. **低温中位数介于两者之间**：median16t05 精度最好（BR −2.59%），但 16 个样本的几何中位数把各样本不同步的摆动混在一起，jerk 1.33×。
5. **随机样本的概率质量没有进步**：三者 ES 都约 .205，仍劣于 OOF bootstrap .128。BR 单样本 dct_mid 是 GT 的 2.25 倍：高 t 配方中的 dct_low/mid 幅度匹配项（原为防止确定性输出塌缩到均值而设）作用在已有随机性的样本上，会把摆动推得过大。
6. **交互（手没到就躲）**：所有采样读出的 primary_excess_antic 都高于 FD2（.0277），BR:ode0 .0330；本轮未改善。

## 后续选项（须用户决定，均需重新预登记或明确授权）

- **接受 mse 取舍**：用户明确授权"以 mpjpe/mae + 全部护栏为准、mse 允许 +5%"之类的修订后，BR:ode0 补 seed1/2 × 10000 并按 3 seed 均值判定；不能事后由我自行放宽。
- **减少押错模式**：例如以 BR 为底座，在 K 个低温样本中选与样本群一致性最高者（medoid）或按 root 走向聚类后取多数簇的 ODE 解，再预登记；或高 t 配方去掉 dct 幅度项后重测单样本。
- **停止扩散路线**：主线保持 FD2。

## 产物

- 代码：`model/forecasting_ntu2p_jitflow.py`（采样温度）、`train/train_ntu2p_jitflow.py`（`recipe_loss`、`--flow_recipe_weight/--flow_recipe_t_min`）、`eval/eval_ntu2p_v2.py`（`--flow_temperature`）、`scripts/diag_ntu2p_band_oracle.py`、`scripts/run_ntu2p_bandflow.py`。
- 检查：τ=1 的 median16 读出与 ReGenNet 存档 1044 项逐位一致；τ=0 采样与 seed 无关（确定性）；K=11 模型首帧残差 1.8e-15、零初始化端点等于 copy-last；上一轮 32 项结构检查全部通过。
- 权重：`save/forecasting/ntu120_label/ntu2p_bandflow_subjval_{B,BR}_s0_10000/ema/model000010000.pt`（训练 B 约 26 min，BR 约 39 min，两组并发）。
