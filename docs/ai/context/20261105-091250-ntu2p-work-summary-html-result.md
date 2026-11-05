<!-- prune:keep -->

# 本轮工作总结页（HTML）结果

计划：`20261105-090726-ntu2p-work-summary-html-plan.md`。用户 2026-11-05："你画个html来说一下你做了什么"。

## 产物

- 私有 Artifact：https://claude.ai/artifact/JATFz1FVwe5Xrw2eyrq5iF （标题"NTU2P 本轮研究记录"）。与架构对比页 https://claude.ai/artifact/Csh1PeSj5inzf5BiHGSghK 互相独立，本页链接过去。
- 本地副本：`results/forecasting/ntu120_label/ntu2p_bandflow/work_summary/index.html` 与 `videos/`（3 段四栏视频及封面）。

## 内容

先说结论（主线 / 确定变好 / 没变好 / 不采纳）→ 流水线改动位置图（采纳：扩散去噪器、11 阶解码、落脚规划；不采纳：关节级交互图，标时间线步号）→ 7 步时间线（用户原话、做了什么、结果、决定）→ 落脚规划与关节级交互图两张机制图 → G1 三个 seed 的 mpjpe 哑铃图（按比例，含判定通过线）→ 原协议 test 上主线 vs FD2 得失表 → 3 段视频（#010 走开、#028 推人、#194 跟随）→ 未解决问题与未提交状态。数字全部取自已有结果文档，没有新计算。

## 检查

发布前用本机 headless Chrome 截图看了一次：表格的说明列与页面容器共用 `.wrap` 类名，行被撑高，已改为 `td.desc`。SVG 中 `fill="var(...)"` 属性会被 CSS 类覆盖，已改为 style。3 段视频描述与此前自己抽帧看到的一致（`jointgraph_s0/videos/*_strip.png`、`compare_page/videos_planner/case_010_*_strip.png`）。
