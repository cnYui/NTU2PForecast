# 细分部位 F13 / 骨架邻接图 SK vs BR（ode0 + 落脚规划，H-val seed2）

|行|mpjpe|xyz_mse|xyz_mae|primary_excess_antic|missed_contact_total|contact_rmse_push|contact_rmse_all|wrong_overlap|slide_frame_ratio|step_count|jerk_body|foot_penetration_ratio|bone_rel_err_body|dct_mid_ratio|gait_sep_corr_f01_10|dct_low_ratio|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|BR|0.1678|0.0198|0.0816|0.0311|48.0000|0.2409|0.1882|0.0034|0.0354|1.7397|22.4145|0.0271|0.0000|0.2826|0.5479|0.6721|
|F13|0.1710|0.0206|0.0830|0.0348|53.0000|0.2387|0.1896|0.0032|0.0335|1.5685|20.8875|0.0256|0.0000|0.3070|0.5755|0.6107|

ode0（不加规划）mpjpe：BR 0.16738，F13 0.17041

F13 判定：无增量
- 增量 精度：mpjpe ≤ 0.992×BR 且 xyz_mse ≤ BR：否
- 增量 交互：超额后退 ≤ 0.9×BR 且漏接触 ≤ BR：否
- 增量 腿：滑行 ≤ BR−0.005、步数 ≥ BR、交替相关 ≥ BR：否
- 护栏 mpjpe ≤ 1.01×BR：未过
- 护栏 jerk ≤ 24.5：过
- 护栏 骨长 ≤ 1e-5：过
- 护栏 穿地 ≤ BR+0.01：过
- 护栏 摆动 gate：过
