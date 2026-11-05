# 落脚规划层（让腿真正迈步）：H-val 结果

设计 `20261104-170558-ntu2p-foot-planner-design-and-plan.md`；修订（摆动时长自适应）`20261104-171132-ntu2p-foot-planner-adaptive-swing-amendment.md`。全部为 H-val，未评 test。

## 决定

- **v1（预登记参数，摆动固定 21 帧）不通过**：抽帧可见后腿被拉成一字马式弓步（见修订文档）。
- **v2（摆动自适应）按预登记的指标判定差一条**：左右交替相关 3 seed 均值 −0.245，要求 ≤ −0.25（逐 seed −0.258 / −0.239 / −0.239）；其余三条步态判定与四条护栏全部通过。按修订文档约定不再调参数；是否接受这条差 0.005 的判定，由用户决定。
- **抽帧目视通过**：9 段固定案例全部逐段看过（`results/forecasting/ntu120_label/ntu2p_bandflow/compare_page/videos_planner/*_strip.png`），步行与后退变成左右交替的离散步、地面足迹成点，原地动作保持静止，无膝反折或腿拉长。

## 3 seed × BR:ode0（H-val，均值 ± 标准差）

汇总：`results/forecasting/ntu120_label/ntu2p_jitflow/subjval/bandflow_foot_planner_summary.{json,md}`（`scripts/summarize_ntu2p_foot_planner.py`）。

| 行 | mpjpe | 滑行帧比（GT .034） | 左右交替相关（GT −.39） | 步数（GT 2.29） | GT 着地帧脚速 | 步相位相关 f1–10 | 穿地 | jerk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FD2 | .1722 | .180 | −.185 | 1.45 | .199 | .468 | .067 | 20.4 |
| FD2 + 规划层 | .1726 | .030 | −.226 | 1.67 | .144 | .538 | .024 | 25.3 |
| BR:ode0 | .1683 ± .0009 | .191 ± .033 | −.081 ± .035 | 1.37 ± .05 | .217 | .465 | .067 | 15.8 |
| **BR:ode0 + 规划层** | .1688 ± .0009 | **.032 ± .003** | −.245 ± .011 | **1.69 ± .04** | **.175** | **.583** | **.028** | 21.8 |

判定（BR:ode0 + 规划层 vs BR:ode0）：滑行 ≤ 0.5× 过；交替相关 ≤ −0.25 **未过**（−0.245）；步数更接近 GT 过；GT 着地帧脚速不高于 BR 过；mpjpe ≤ 1.01× 过（+0.3%）；骨长、穿地、jerk ≤ 1.2×FD2（24.5）过。

## 诊断要点

1. **表示不是瓶颈**（`scripts/diag_ntu2p_gait_oracle.py`）：GT 经 11 阶 DCT 编解码仍迈步（步数 2.22、交替相关 −0.40）；FD2 与 BR:ode0 的预测左右脚几乎不交替，脚随骨盆平移（滑行帧比为 GT 的 6 倍）。确定性读出把步相位平均掉。
2. **GT 接触判定会严重高估摆动时长**：NTU 脚部拟合抖动使接触阈值口径的摆动中位 21 帧；不依赖阈值的口径（踝速 > 1.5×root）中位 7 帧，与踝峰速/root 均速 4.28 一致。用 21 帧导致 v1 的弓步。
3. 规划层与模型无关：FD2 加规划层同样把滑行从 .180 降到 .030。

## 边界

- 规划层是运动学投影，不是学到的步态：何时迈步由"脚被身体带开多远"决定，不保证与 GT 的步相位一致（步相位相关 .58，GT 为 1）。它把漂移转换成离散的步，不会补出模型没预测出来的位移，也不改模型选错的走向（194 号窗 B 走进 A 的位置、216 号窗过宽站姿仍在）。
- 交互时序（"手没碰到就躲"）不受影响（primary_excess_antic 不变）。
- jerk 上升（15.8 → 21.8）来自真实摆动带来的快速脚部运动，仍在护栏内。

## 产物

- `utils/ntu2p_foot_planner.py`（`plan_feet`）；`eval/eval_ntu2p_v2.py --foot_planner`。
- 诊断：`scripts/diag_ntu2p_gait_oracle.py`、`scripts/diag_ntu2p_gt_gait_stats.py`；汇总 `scripts/summarize_ntu2p_foot_planner.py`；抽帧 `scripts/extract_video_strip.py`。
- 四栏视频（GT | FD2 | BR:ode0 | BR:ode0 + 规划层）：`results/forecasting/ntu120_label/ntu2p_bandflow/compare_page/videos_planner/`。
