<!-- prune:keep -->

# 动画架构图页面结果

计划：`20261105-093627-ntu2p-animated-architecture-page-plan.md`。

## 产物

- 私有 Artifact：https://claude.ai/artifact/S8SuoBo8xa2SrL9bcgw74E （"NTU2P 两代模型架构"）。
- 本地：`results/forecasting/ntu120_label/ntu2p_bandflow/arch_page/`（`index.html` 为发布版；`template.html` 是带 `__DEMO_DATA__` 占位的源；`demo.json` 由 `scripts/export_ntu2p_arch_demo.py` 生成）。

## 页面内容

1. 输入与输出：#028 推人、#010 走开两个 H-val 样本，三栏同步播放 GT | FD2 | 主线（BR:ode0 + 落脚规划），可拖动时间条；前 10 帧是三栏相同的输入。张量说明：`obs_xyz [B,10,2,55,3]` + `action [B]` → `pred_xyz [B,50,2,55,3]`。
2. 扩散主线流程图（15 步）：数据包沿各级移动，进入采样循环后绕 19 圈、t 从 0 走到 1；右侧面板逐步说明并给出张量形状，小图同步播放当前 t 的状态 z 解码出的动作（t = 0 时 z = 0，即两人保持观测末帧不动），最后切到加落脚规划的输出。
3. FD2 流程图（14 步）：一次前向，InterMixer 三个子步骤与 ×6 回路，输出头、腿流、刚体手组装。
4. 对照表：逐环节相同/替换，参数量与 test 指标。

## 核对

- 结构与形状逐项对照 `model/forecasting_ntu2p_intermixer.py`、`model/forecasting_ntu2p_jitflow.py`、`utils/ntu2p_foot_planner.py` 与两个 seed0 checkpoint 的配置（FD2：hidden 256、6 块、input_dct_k 30、root 8、local 20、刚体手、腿流、骨架投影、3.65M；BR：STG 宽 192、4 头、6 块、patch 5、root 8、local 11、自条件、无跨人因果、4.57M；Heun 20 步共 39 次前向）。
- 状态 z 用 BR seed0 在 CPU 上按 ode0 重算：最终状态与对比页导出的 BR:ode0（GPU、分批 16）最大差 0.66 mm（#028）/ 1.20 mm（#010）；20 步内 x̂ 的估计最大变化约 37 cm，说明采样过程中估计确实在变。
- 发布前本机 headless Chrome 截图一次：修正数据包圆点压住框内文字（改为沿框左边缘移动）、回流线与竖排说明文字过近。脚本语法用 node 检查通过。
- 频率表述：主线局部 11 阶写"≤ 2 Hz"（60 帧窗口 DCT 第 10 阶约 1.7 Hz）；FD2 的 20 阶不再写具体 Hz（对比页曾写"到 3.8 Hz"，按 60 帧窗口约为 3.2 Hz）。
