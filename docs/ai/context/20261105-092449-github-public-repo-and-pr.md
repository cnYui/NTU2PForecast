<!-- prune:keep -->

# 发布到 GitHub：公开仓库与首个 PR

用户 2026-11-05："你把代码提pr往github提交一下"。确认时用户选择**新建公开仓库** `cnYui/NTU2PForecast`，并同意 ReGenNet 上未提交的 JiT-RL 工作按其 `ship_pr.py` 流程一并提交、自动合并（记录在 ReGenNet 的同名时间戳文档）。

## 做法

1. `main` 只含提取时的初始提交 `f21770a`，再加一个只放 `LICENSE` 的提交：提取的文件里有 6 个来自上游 ReGenNet（`model/rotation2xyz.py`、`model/smpl.py`、`utils/config.py`、`utils/fixseed.py`、`utils/rotation_conversions.py`、`environment.yml`），上游为 MIT（Copyright (c) 2024 Liang Xu），公开发布须附许可证原文；`utils/PYTORCH3D_LICENSE` 已随提取带入。
2. `gh repo create cnYui/NTU2PForecast --public`，推送 `main`。
3. 2026-11-03 起的全部工作放在 `feat/ntu2p-diffusion-mainline` 分支一次提交，向 `main` 提 PR；本仓库没有自动合并 CI，由用户审阅后合并。

## 发布前检查

- 待发布 279 个文件约 4.6 MB：代码、`docs/ai/context`、结果 json/md/log、对比页与总结页 html、视频封面 jpg。权重（`save/`、`*.pt`）、`*.png`、`*.mp4` 与指向 ReGenNet 的数据软链接（`dataset/`、`body_models/`、两个 xyz 缓存）按 `.gitignore` 不入库；manifest 只含 NTU 样本编号，不含骨架数据。
- 扫描密钥（GitHub/OpenAI/AWS token、私钥、`api_key/password/secret`）与邮箱地址：无命中。文档里有本机绝对路径（含用户名 `rpartx3080`），用户选择公开时已知。
- 全部 `.py` 在项目环境 Python 3.7.13 下 `py_compile` 通过。
