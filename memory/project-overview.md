---
name: ocr-reasoning-project-overview
description: OCR-Reasoning 仓库的稳定结构、数据流、运行约定和协作注意事项
metadata:
  type: project
  last_verified: 2026-08-18
  commit: "6282752"
---

# 项目定位

OCR-Reasoning 是基于 VLMEvalKit 二次开发的文字密集型图像推理研究仓库。核心目标不只是评测最终答案，还包括研究视觉定位、信息抽取、分步推理、依赖传播和最终综合中的错误来源。

基准主体有 1,069 条人工标注样本，覆盖空间、数值、数学、枚举、逻辑和跨学科推理。当前研究代码还从其中构造了约 200 题的 Step-QA 实验数据；“1,069 条原始基准”和“约 200 题分步实验子集”不要混为一谈。

# 工作区组成

- `OCR-Reasoning-main/`：核心 Git 仓库，包含 VLMEvalKit、OCR agents、数据集、评测与实验脚本。
- `RIG-Annotating-main/`：独立 PyQt5 框选标注工具。JSON 项目、30 秒自动保存、图片与矩形标注管理位于 `core/` 和 `ui/`。
- `examination_demo/`：独立 PyQt5 多语言阅卷界面原型；OCR/VLM/语言识别后端仍以 TODO 接口为主。

除非任务明确涉及 GUI，默认工作目录应是 `OCR-Reasoning-main/`。

# 核心执行链

1. `run.py` 解析 `--data`、`--model`、`--mode`、`--judge` 和输出目录。
2. `vlmeval/config.py` 的 `supported_VLM` 用注册名构造本地模型或 OpenAI-compatible API wrapper。
3. `vlmeval/dataset/__init__.py` 构造数据集；原始基准实现是 `vlmeval/dataset/image_vqa.py::OCR_Reasoning`。
4. `OCR_Reasoning.build_prompt()` 加入中英文逐步推理与最终答案格式要求。
5. 推理结果写到 `outputs/<model>/<run-id>/` 一类目录；具体层级由 `run.py` 生成。
6. `OCR_Reasoning.evaluate()` 调用 `vlmeval/dataset/utils/ocr_reasoning.py` 中的 `OcrR_auxeval` 和 `OcrR_acc`，需要可工作的 judge API。

评测侧的一个重要实现细节：`OCR_Reasoning.evaluate()` 当前把 `nproc` 强制设为 1，即使命令行传入更高并行度，judge 阶段仍是串行的。

# 关键目录与文件

- `run.py`：统一推理/评测入口。无参数运行不可靠，因为 `--model` 使用 `nargs='+'`，但默认值是字符串；始终显式传 `--model`。
- `vlmeval/config.py`：模型注册与实验参数的最终来源。包含内网 API、绝对 Windows 路径和本地模型配置，移植时需覆盖而非照搬。
- `vlmeval/vlm/ocr_agent_v4.py`、`v5.py`、`v6.py`：当前主要 agent 演化线。
- `vlmeval/dataset/step_reasoning_qa.py`：`Step_Reasoning_QA`、`OCR_Reasoning_StepQA`、`Depended_Step_Reasoning_QA`。
- `step_qa_pipeline.py` / `step_qa_generate.py` / `verify_step_qa.py`：分步 QA 的生成与人工/脚本验证。
- `build_step_reasoning_qa_tsv.py`、`build_depended_step_qa.py`、`build_depended_tsv.py`：把 Excel 中间产物构造成 VLMEvalKit 可加载的 TSV。
- `evaluate_step_reasoning_qa.py`、`evaluate_depended_serial.py`：分步与依赖式评测的独立入口。
- `test_case/`：小规模复现、图片、bbox、trace 和诊断产物，不等价于规范化单元测试目录。
- `outputs/`：实验结果和派生数据，部分脚本直接依赖这里的 Excel/TSV。
- `local_*`：本地模型快照或配置，体积大且机器相关。

# 常用命令

从 `OCR-Reasoning-main/` 执行，并明确指定模型：

```powershell
python run.py --data OCR_Reasoning --model Qwen3.5-9B-OCRAgentV5 --verbose
python run.py --data OCR_Reasoning --model Qwen3.5-9B-OCRAgentV6 --mode infer --verbose
python run.py --data Step_Reasoning_QA --model Qwen3.5-9B-API --judge qwen-plus
python run.py --data Depended_Step_Reasoning_QA --model Qwen3.5-9B-API --judge qwen-plus
```

只做推理用 `--mode infer`；完整评测会调用 judge。`--reuse` 会复用最近结果，调试 agent 行为时需确认是否希望复用旧预测。

# 环境与可移植性

- 根 `requirements.txt` 是宽松依赖集合，没有锁定可复现实验版本；`setup.py` 声称 Python `>=3.7`，但新脚本使用了 `list[str]` 等语法，实际应使用 Python 3.9+，优先 3.10。
- 工作区能看到 `ocrr`、`ocrr-old`、`ocrr-T`、`finesplat` 等 Conda 环境，但未形成仓库级唯一环境约定；执行前检查目标环境中的 `torch`、`transformers`、`openpyxl`、`pandas`、`Pillow` 等依赖。
- 数据默认由 VLMEvalKit/Hugging Face 下载到 `LMUData`，可用 `LMU_DATA` 改根目录。部分 Step-QA 代码仍含本机绝对候选路径，应优先保留相对路径 fallback。
- API 通过 `.env`、环境变量或 `vlmeval/config.py` 中的注册参数提供。不得在记忆、日志摘要或提交中复制真实密钥。

# 修改与验证约定

- agent 行为修改后，至少检查：schema 是否同步、注册项是否仍能构造、`generate_inner()` 的失败路径、trace 字段、最终 `<a>...</a>` 格式。
- 数据集修改后，检查本地 TSV/Excel 路径、`index` 类型、图片绝对路径、语言字段和 judge 输出列。
- 优先用针对性的 smoke script；仓库中大量 `test_*.py` 是依赖模型服务/大数据的实验脚本，不适合一次性全跑。
- 输出与失败日志可能非常多。不要用宽泛的删除或 Git 恢复命令处理它们；先精确确认目标和用户现有改动。
