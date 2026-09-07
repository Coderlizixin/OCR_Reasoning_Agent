# OCR-Reasoning 项目记忆

> 最近核对：2026-09-05，`main` 分支，提交 `b9914e3`。

- [项目总览](project-overview.md) — 仓库定位、核心数据流、目录职责、运行与评测约定
- [当前研究与实验状态](current-research.md) — OCR-Agent V4/V5/V6、Step-QA、已知漂移与排障入口
- [OCR-Agent V4](ocr-agent-v4.md) — V4 的 Planning → Executor(DAG) → Reasoning 设计细节

## 使用原则

- 先读“项目总览”，涉及 agent 或逐步评测时再读对应专题。
- `vlmeval/config.py` 和实际实现代码是配置/行为的最终事实来源；注释、旧记忆和空的 `README.md` 不能单独作为依据。
- 仓库常含实验输出、图片、trace 和失败请求日志。开始工作前先检查 `git status`，不要清理、恢复或覆盖与当前任务无关的改动。
- API 地址、本地模型目录、数据目录和密钥属于机器/实验配置；不要把新值固化进项目记忆，也不要提交 `.env`。
