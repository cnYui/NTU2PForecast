<!-- prune:keep -->

# 跨人腿部 Transformer（XL）分数对比页

用户 2026-11-06："你把这个结果也成一个html，用来比较原始模型分数和当前分数"。

- 私有 Artifact：https://claude.ai/artifact/WFDyB69HMLRN81igiptJGV （"跨人腿部 Transformer 对照"）；本地 `results/forecasting/ntu120_label/ntu2p_bandflow/legx_compare_page/`。
- "原始模型"取主线 BR，"当前"为 XL；另列原始 DCT 模型 FD2 作参照（`bandflow_s0/FD2ref_plan.json`，同样加落脚规划；其错误重叠帧比无导出数组，记为 —）。全部为 H-val seed0、ode0 + 落脚规划。
- 内容：判定标签与四个关键数（mpjpe +1.94%、超额后退 +53%、漏接触 +19%、左右交替 −12%）；14 项指标相对 BR 的百分比变化图（按越低/越高越好标绿红，FD2 以空心点叠加）；原始分数表；预登记增量与护栏逐条；#028、#194 抽帧条带。数字取自 `subjval/legx_s0/summary.json`，无新计算。结果文档 `20261105-225809-ntu2p-cross-person-leg-transformer-result.md` 中 mpjpe 写作 +1.95%，按原始值 .16959 → .17289 为 +1.945%，页面四舍五入为 +1.94%。
- 发布前截图看过一次。
