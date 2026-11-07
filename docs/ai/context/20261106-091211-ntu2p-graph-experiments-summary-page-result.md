<!-- prune:keep -->

# 图结构实验汇总页：结果

计划：`20261106-091007-ntu2p-graph-experiments-summary-page-plan.md`。

- 私有 Artifact：https://claude.ai/artifact/8zAwDp7tnVTZRG6Q8jfrhz （"NTU2P 图结构实验汇总"）；本地 `results/forecasting/ntu120_label/ntu2p_bandflow/graph_summary_page/index.html`。
- 收录五次尝试：STG 时空图注意力 J3（主线在用；3 seed −3.30% vs FD2，但 J4N 只有 −0.71%，收益主要来自流训练）、接触图纠正器 C2/F2（vs C1/F1：+0.04% / −0.013%）、J4 接触图 + 跨人因果（vs J3：mpjpe +0.08%，超额后退 +21%）、关节级交互图 G1/G2（G1 3 seed −0.35%、seed1/2 +0.60%；交互判定不过）、跨人腿部 Transformer XL（+1.94%，超额后退 +53%）。G2 seed0 精确值 mpjpe .16967 vs BR .16959（+0.05%）。
- 对用户两个问题的判断写在页面：
  1. 经典骨架 GCN：优先级低。人体内部结构已有骨架投影、刚体手与部位 token；剩余问题源于确定性读出平均多模态未来，而非关节间信息传递；跨人边仍需人为设计（G1 已试）。若要确认，可在 STG 注意力上加骨架邻接偏置/掩码，seed0 约 1 小时。
  2. 部位再细分：G1 的 13 组同时叠加了距离边与跨人因果，"只细分"从未干净地测过。建议对照：`part_scheme="fine"`，不加 `joint_graph`、不加 `causal_cross`，其余同 BR，H-val seed0 × 10000，判定事先写定。每关节一个节点（696 token，注意力约 34×）不建议。
- 发布前截图一次，修正示意图标注压线与图表右侧数字被截断。
