# 相位流形步态模块 v4a 第 0 阶段（H-val seed0）

## 诊断（只换腿）

|行|step_count|slide_frame_ratio|stance_fraction|lr_forward_vel_corr|skate_gt_contact_speed_walk|mpjpe|leg_err|
|---|---:|---:|---:|---:|---:|---:|---:|
|GT|2.2877|0.0344|0.3478|-0.3898|0.1004|0.0000|0.0000|
|S0-1 GT 相位 + GT root|2.1781|0.0396|0.3566|-0.3632|0.1492|0.0048|0.0438|
|S0-2 预测相位 + GT root|2.2740|0.0401|0.3140|-0.2667|0.2534|0.0092|0.0843|
|BR:ode0 腿（参照）|1.3904|0.1988|0.4450|-0.0685|0.2062|0.1691|0.1273|
|BR:ode0 的腿接到 GT root（参照）|1.3904|0.3166|0.3335|0.0990|0.2377|0.0103|0.0943|

## S0-3 完整评价

|行|mpjpe|xyz_mse|xyz_mae|primary_excess_antic|missed_contact_total|contact_rmse_push|contact_rmse_all|wrong_overlap|slide_frame_ratio|step_count|jerk_body|foot_penetration_ratio|bone_rel_err_body|dct_mid_ratio|gait_sep_corr_f01_10|dct_low_ratio|stance_fraction|lr_forward_vel_corr|skate_gt_contact_speed_walk|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|BR:ode0|0.1691|0.0202|0.0820|0.0330|43.0000|0.2070|0.1918|0.0049|0.1988|1.3904|15.3733|0.0680|0.0000|0.2321|0.4601|0.6265|0.4450|-0.0685|0.2062|
|PGait|0.1698|0.0202|0.0824|0.0330|43.0000|0.2070|0.1918|0.0061|0.0330|2.0548|20.6354|0.0202|0.0000|0.2577|0.4423|0.6332|0.3556|-0.2157|0.2930|

判定：可替代落脚规划层
- 护栏 mpjpe ≤ 1.005×BR：过
- 护栏 xyz_mse ≤ 1.02×BR：过
- 护栏 jerk ≤ 24.5：过
- 护栏 穿地 ≤ BR+0.01：过
- 护栏 骨长 ≤ 1e-5：过
- 护栏 摆动 gate：过
- 护栏 超额后退 ≤ 1.1×BR：过
- 护栏 漏接触 ≤ 1.1×BR：过
