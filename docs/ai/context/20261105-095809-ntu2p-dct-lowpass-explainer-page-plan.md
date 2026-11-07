# DCT 低通去抖说明页计划

用户 2026-11-05："你把上面说的这个公式和过滤的方案写成一个新的html"（上文：DCT 为什么能滤掉抖动）。

## 内容

1. 交互演示：4 条真实 GT 关节轨迹（H-val #028 推人：A 右手腕高度=真实推手、A 左手腕高度=遮挡跳点、A 左踝高度=站立不动的脚；#010 走开：A 左踝高度=迈步），60 帧（观测 10 + 未来 50），滑块选保留阶数 K（默认 11）。三张图：原始 vs 重建、被去掉的部分、频谱 c_k² 与截断线（横轴同时标 f_k = k/6 Hz）。数字：最高保留频率、去掉的能量占比（不计 k=0）、去掉部分 RMS。
2. 公式：正交 DCT-II、频率 f_k = k f_s / 2N、截断重建、Parseval 误差。
3. 模型里的用法：只取未来 50 帧、乘 ramp、QR 正交化；root 8 阶、局部 11 阶；为什么对扩散关键（jerk 55 → 29）。
4. 代价：截断丢掉真实快动作；GT 经 11 阶编解码的代价为 FD2 mse 的 0.82%。

演示用的是普通 60 帧 DCT（无 ramp），页面说明这一点。数据取自 `results/forecasting/ntu120_label/ntu2p_bandflow/arch_page/demo.json`（规范系，毫米）。公式用 cdnjs MathJax 3.2.2（tex-svg-full）。本地 `results/forecasting/ntu120_label/ntu2p_bandflow/dct_page/`。
