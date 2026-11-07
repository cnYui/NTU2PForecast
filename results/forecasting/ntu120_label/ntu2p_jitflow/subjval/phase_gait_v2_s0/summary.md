# 相位流形步态模块 v2 第 0 阶段（H-val seed0）

## 诊断（只换腿）

|行|step_count|slide_frame_ratio|stance_fraction|lr_forward_vel_corr|skate_gt_contact_speed_walk|mpjpe|leg_err|
|---|---:|---:|---:|---:|---:|---:|---:|
|GT|2.2877|0.0344|0.3478|-0.3898|0.1004|0.0000|0.0000|
|S0-1 GT 相位 + GT root|2.1507|0.0410|0.3379|-0.3758|0.1612|0.0057|0.0526|
|S0-2 预测相位 + GT root|2.2397|0.0396|0.2944|-0.3182|0.2672|0.0099|0.0904|
|BR:ode0 腿（参照）|1.3904|0.1988|0.4450|-0.0685|0.2062|0.1691|0.1273|
|BR:ode0 的腿接到 GT root（参照）|1.3904|0.3166|0.3335|0.0990|0.2377|0.0103|0.0943|

## S0-3 完整评价

|行|mpjpe|xyz_mse|xyz_mae|primary_excess_antic|missed_contact_total|contact_rmse_push|contact_rmse_all|wrong_overlap|slide_frame_ratio|step_count|jerk_body|foot_penetration_ratio|bone_rel_err_body|dct_mid_ratio|gait_sep_corr_f01_10|dct_low_ratio|stance_fraction|lr_forward_vel_corr|skate_gt_contact_speed_walk|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|BR:ode0|0.1691|0.0202|0.0820|0.0330|43.0000|0.2070|0.1918|0.0049|0.1988|1.3904|15.3733|0.0680|0.0000|0.2321|0.4601|0.6265|0.4450|-0.0685|0.2062|
|PGait|0.1703|0.0202|0.0826|0.0330|43.0000|0.2070|0.1918|0.0062|0.0334|2.0822|26.2664|0.0208|0.0000|0.2551|0.4083|0.6319|0.3661|-0.3196|0.2889|

判定：无效（护栏未过）；步态条件达标（可替代落脚规划层级）
- 护栏 mpjpe ≤ 1.005×BR：未过
- 护栏 xyz_mse ≤ 1.02×BR：过
- 护栏 jerk ≤ 24.5：未过
- 护栏 穿地 ≤ BR+0.01：过
- 护栏 骨长 ≤ 1e-5：过
- 护栏 摆动 gate：过
- 护栏 超额后退 ≤ 1.1×BR：过
- 护栏 漏接触 ≤ 1.1×BR：过
