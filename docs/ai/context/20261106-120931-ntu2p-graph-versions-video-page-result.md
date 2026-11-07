<!-- prune:keep -->

# 各图结构版本视频对照页：结果

计划：`20261106-120506-ntu2p-graph-versions-video-page-plan.md`。

- 私有 Artifact：https://claude.ai/artifact/VHLX6RKEoCFQNJ6g8XunZ6 （"图结构各版本视频对照"）；本地 `results/forecasting/ntu120_label/ntu2p_bandflow/graph_videos_page/`（`combined_arrays.pt` 六版本合并数组、`case_mpjpe.json`、`page_data.json`、`videos_a/`、`videos_b/`）。
- 六版本（H-val seed0，窗口顺序与 GT 已逐位核对一致）：FD2（原样，无落脚规划）、BR、F13、G1、SK、XL（后五个为 ode0 + 落脚规划）。每例两段同步四栏视频：GT | FD2 | BR | F13 与 GT | G1 | SK | XL。
- 逐例 mpjpe（FD2 / BR / F13 / G1 / SK / XL）：#028 .187/.199/.181/.185/.181/.176；#002 .149/.140/.151/.143/.158/.143；#194 .314/.333/.384/.333/.331/.398；#010 .131/.175/.198/.208/.195/.180；#098 .288/.310/.292/.287/.295/.289；#216 .152/.181/.160/.169/.216/.162。
- 抽帧（新看的两例）：#216 一起搬东西，F13 站姿正常、比 BR 接近 GT；BR 与 SK 的 B 出现很宽弓步，SK 的 A 后仰。#098 拥抱，所有扩散版本 A 两腿分得偏开，F13 在 2.3 s A 的后腿伸得很远，FD2 站姿最直。其余四例沿用此前抽帧结论（#194 F13/XL 两人重叠交叉更糟，#010 F13 单腿大跨步）。
- 页面总分表 G1 一行的 −2.23% 加注为 seed0 偶然结果（补 seed 后 +0.60%）。F13 的 seed1/2 验证仍在训练（截至 12:09 约第 1500 步），结论待定。
