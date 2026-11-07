# 细分部位 F13 / 骨架邻接图 SK vs BR（ode0 + 落脚规划，H-val seed1）

|行|mpjpe|xyz_mse|xyz_mae|primary_excess_antic|missed_contact_total|contact_rmse_push|contact_rmse_all|wrong_overlap|slide_frame_ratio|step_count|jerk_body|foot_penetration_ratio|bone_rel_err_body|dct_mid_ratio|gait_sep_corr_f01_10|dct_low_ratio|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|BR|0.1690|0.0201|0.0821|0.0282|42.0000|0.2443|0.1918|0.0070|0.0319|1.6712|21.8917|0.0296|0.0000|0.3232|0.5210|0.6904|
|F13|0.1712|0.0212|0.0828|0.0396|51.0000|0.2014|0.1973|0.0057|0.0299|1.5890|21.5826|0.0257|0.0000|0.2751|0.5466|0.6237|

ode0（不加规划）mpjpe：BR 0.16856，F13 0.17073

F13 判定：无增量
- 增量 精度：mpjpe ≤ 0.992×BR 且 xyz_mse ≤ BR：否
- 增量 交互：超额后退 ≤ 0.9×BR 且漏接触 ≤ BR：否
- 增量 腿：滑行 ≤ BR−0.005、步数 ≥ BR、交替相关 ≥ BR：否
- 护栏 mpjpe ≤ 1.01×BR：未过
- 护栏 jerk ≤ 24.5：过
- 护栏 骨长 ≤ 1e-5：过
- 护栏 穿地 ≤ BR+0.01：过
- 护栏 摆动 gate：过
