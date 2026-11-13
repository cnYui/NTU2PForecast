# 相位流形步态模块 v3 第 0 阶段（H-val seed0）

## 诊断（只换腿）

|行|step_count|slide_frame_ratio|stance_fraction|lr_forward_vel_corr|skate_gt_contact_speed_walk|mpjpe|leg_err|
|---|---:|---:|---:|---:|---:|---:|---:|
|GT|2.2877|0.0344|0.3478|-0.3898|0.1004|0.0000|0.0000|
|S0-1 GT 相位 + GT root|2.1644|0.0382|0.3639|-0.3635|0.1469|0.0048|0.0438|
|S0-2 预测相位 + GT root|2.2534|0.0426|0.3149|-0.3194|0.2626|0.0094|0.0862|
|BR:ode0 腿（参照）|1.3904|0.1988|0.4450|-0.0685|0.2062|0.1691|0.1273|
|BR:ode0 的腿接到 GT root（参照）|1.3904|0.3166|0.3335|0.0990|0.2377|0.0103|0.0943|

## S0-3 完整评价

|行|mpjpe|xyz_mse|xyz_mae|primary_excess_antic|missed_contact_total|contact_rmse_push|contact_rmse_all|wrong_overlap|slide_frame_ratio|step_count|jerk_body|foot_penetration_ratio|bone_rel_err_body|dct_mid_ratio|gait_sep_corr_f01_10|dct_low_ratio|stance_fraction|lr_forward_vel_corr|skate_gt_contact_speed_walk|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|BR:ode0|0.1691|0.0202|0.0820|0.0330|43.0000|0.2070|0.1918|0.0049|0.1988|1.3904|15.3733|0.0680|0.0000|0.2321|0.4601|0.6265|0.4450|-0.0685|0.2062|
|PGait|0.1700|0.0202|0.0824|0.0330|43.0000|0.2070|0.1918|0.0049|0.0312|2.0959|20.9691|0.0225|0.0000|0.2598|0.4491|0.6359|0.3590|-0.3406|0.2996|

判定：无效（护栏未过）；步态条件达标（可替代落脚规划层级）
- 护栏 mpjpe ≤ 1.005×BR：未过
- 护栏 xyz_mse ≤ 1.02×BR：过
- 护栏 jerk ≤ 24.5：过
- 护栏 穿地 ≤ BR+0.01：过
- 护栏 骨长 ≤ 1e-5：过
- 护栏 摆动 gate：过
- 护栏 超额后退 ≤ 1.1×BR：过
- 护栏 漏接触 ≤ 1.1×BR：过
