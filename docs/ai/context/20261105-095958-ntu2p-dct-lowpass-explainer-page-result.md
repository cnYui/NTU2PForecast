<!-- prune:keep -->

# DCT 低通去抖说明页结果

计划：`20261105-095809-ntu2p-dct-lowpass-explainer-page-plan.md`。

- 私有 Artifact：https://claude.ai/artifact/M6a3hHRJzYVb72twm9cXP5 （"DCT 为什么能滤掉抖动"）；本地 `results/forecasting/ntu120_label/ntu2p_bandflow/dct_page/`（`template.html` + `series.json` 生成 `index.html`）。
- 4 条真实 GT 轨迹（高度轴，规范系）在 K=11 时去掉的能量（不计均值）：推人 A 右手腕（真实推手，起伏 885 mm）0.1%、高频 RMS 约 7 mm；推人 A 左手腕（遮挡跳点）约 19%、RMS 约 50 mm；推人 A 左踝（站立）约 32%、RMS 约 3 mm；走开 A 左踝（迈步，起伏 229 mm）0.4%、RMS 约 5 mm。即真实动作几乎全保留，跳点和静止脚的起伏大比例被去掉。
- 页面频率按 60 帧窗口 \(f_k=k/6\) Hz 给出（11 阶最高 1.67 Hz，20 阶 3.17 Hz），更正对比页"20 阶到 3.8 Hz"的写法。
- 发布前 headless Chrome 截图一次，MathJax（cdnjs 3.2.2）与三张图正常。
