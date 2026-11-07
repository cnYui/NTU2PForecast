<!-- prune:keep -->

# 架构图页面新增"脚怎么钉在地面上"一节

用户 2026-11-05："你这里说的每只脚它是怎么固定在地面上的，这个你也写入到两代模型架构的那个文档里"。页面：https://claude.ai/artifact/S8SuoBo8xa2SrL9bcgw74E （第 2 版），本地 `results/forecasting/ntu120_label/ntu2p_bandflow/arch_page/`（`template.html` 已同步）。

新增内容对照 `utils/ntu2p_foot_planner.py` 的 `_plan` 与 `_two_bone_ik` 写成：地面高度 = 观测 10 帧踝的最低高度；起始时末帧踝高差 < 0.05 m 即着地；着地期间逐帧把踝和脚掌直接写成锁定点（输出速度为 0）；抬脚条件（水平漂移 > 0.16 m、站稳 ≥ 5 帧、另一只脚着地，同时满足时漂得远的先走）；摆动帧数 \(S=\mathrm{clamp}(\mathrm{round}(20\ell/(2.85v)),5,12)\)、落点 = 第 \(t+S-1\) 帧预测踝贴地后前移 0.10 m；轨迹 smoothstep + 0.07·sin(πu) 抬高；自由状态阈值 0.20 m / 重新着地 0.05 m；两骨 IK（髋取预测，骨长取观测末帧，膝取解圆上离原预测膝最近点，够不着时踝取最大可达）。另在主线流程图第 ⑨ 步说明里指向这一节，页面加载 MathJax（cdnjs 3.2.2）显示公式。发布前截图看过一次。
