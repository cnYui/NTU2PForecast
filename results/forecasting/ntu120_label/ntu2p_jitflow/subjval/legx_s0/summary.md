# 跨人腿部 Transformer XL vs BR（ode0 + 落脚规划，H-val seed0）

|行|mpjpe|xyz_mse|xyz_mae|primary_excess_antic|missed_contact_total|contact_rmse_push|contact_rmse_all|wrong_overlap|slide_frame_ratio|step_count|jerk_body|foot_penetration_ratio|bone_rel_err_body|dct_mid_ratio|gait_sep_corr_f01_10|dct_low_ratio|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|BR|0.1696|0.0203|0.0822|0.0330|43.0000|0.2070|0.1918|0.0059|0.0300|1.6575|21.1620|0.0266|0.0000|0.2680|0.6807|0.6382|
|XL|0.1729|0.0216|0.0838|0.0504|51.0000|0.2221|0.2018|0.0064|0.0353|1.6301|19.2015|0.0246|0.0000|0.3016|0.5980|0.6687|

ode0（不加规划）mpjpe：BR 0.16911，XL 0.17233

判定：无增量
- 增量 精度：mpjpe ≤ 0.992×BR 且 xyz_mse ≤ BR：否
- 增量 交互：超额后退 ≤ 0.9×BR 且漏接触 ≤ BR：否
- 增量 腿：滑行 ≤ BR−0.005、步数 ≥ BR、交替相关 ≥ BR：否
- 护栏 mpjpe ≤ 1.01×BR：未过
- 护栏 jerk ≤ 24.5：过
- 护栏 骨长 ≤ 1e-5：过
- 护栏 穿地 ≤ BR+0.01：过
- 护栏 摆动 gate：过
