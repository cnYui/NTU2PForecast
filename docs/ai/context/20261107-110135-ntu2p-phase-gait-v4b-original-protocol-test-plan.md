<!-- prune:keep -->

# PGait v4b 原协议 test 计划

用户 2026-11-07："评一次 test"。依据：`20261107-105943-ntu2p-phase-gait-v4-3seed-result.md`，H-val 3 seed 全部通过 v4 预登记。

## 做法

1. **训练**：在原协议 train 上训练步态模块，seed 0/1/2。
   - 缓存：`ntu2p_xyz_seq_cache`；manifest：`ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json`。
   - 参数与 H-val 的 v4b 完全相同：`--ground_aware --penetration_weight 10 --obs_recon_weight 1.0 --f_steps 12000 --obs_pose_input --anchor_tau 4 --vel_lowpass_order 8 --leg_l2_weight 3.0`。
   - 只用 train split，不看 val 或 test。
2. **评价**：模块 seed s 挂在原协议 BR seed s 上（`ntu2p_bandflow_original_BR_s{s}_10000/ema`），读出 ode0，在 test（1253 条）上**每个 seed 只评一次**。
   - 命令与主线 test 相同，加 `--gait_head`，导出数组。
3. **对照**：已有的同命令 test 结果 BR:ode0 与 FD2（`results/forecasting/ntu120_label/ntu2p_jitflow/original/bandflow_test/`），不重评。
4. **报告内容**：
   - 3 seed 均值 ± 标准差；
   - 原预登记判定在 test 上的结果（mpjpe ≤ 1.005×、滑行 ≤ .06、步数 ≥ 1.60，以及全部护栏）；
   - 逐 seed 差值；
   - 抽帧目视。
   test 只报告，不据此再选型、调参或重训。

输出：`results/forecasting/ntu120_label/ntu2p_jitflow/original/phase_gait_test/`。驱动：`scripts/run_ntu2p_phase_gait_test.py`。
