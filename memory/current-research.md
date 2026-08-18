---
name: ocr-reasoning-current-research
description: 当前 OCR agent 与 Step-QA 研究线、实现差异和已知陷阱
metadata:
  type: project
  last_verified: 2026-08-18
  commit: "6282752"
---

# 当前主线

最近提交的重心是两条相互关联的研究线：

1. OCR-Agent 的 Planning → Executor → Reasoning 分阶段设计，重点分析 grounding/understanding 对最终推理的影响。
2. 从 OCR-Reasoning 推理过程构造 Step-QA 和依赖式 Step-QA，用于区分单步能力、依赖传播和最终综合错误。

最近一次提交还包含 CharXiv grounding 分析，因此 `test_RIG_on_Charxiv.py`、`_test_descriptive_q1.py`、`test_case/RIG_Charxiv/` 及 RIG 标注工具属于当前诊断上下文。

# OCR-Agent 版本关系

## V4

- 四类步骤：`ground-understand`、`transcribe`、`compute`、`logical`。
- Planning 输出 strategy 与满足“依赖只能指向更早步骤”的 DAG。
- Executor 按顺序/依赖执行；每阶段最多 3 次尝试。
- `ground-understand` 可走 locate → crop → understand，也可由 `skip_locate=True` 直接看整图抽取。
- 顶层 visual step 可做交叉验证；V4 仍保留失败 visual step 的 executor 层 re-plan。
- Reasoning 区分视觉观察（FACT）和计算/逻辑派生（DRAFT），输出 JSON 后格式化为推理文本与 `<a>answer</a>`。

## V5

- 研究意图是把失败后的 executor re-plan 改为 planning 层拆分，并放松计划粒度。
- 注册的 9B/27B V5 默认 `skip_locate=True`、`cross_validate=True`，所以常用 V5 实际绕过 bbox locate；要研究显式定位，使用 `Qwen3.5-9B-OCRAgentV5-Grounding-Understanding`（`skip_locate=False`）。
- 源码/配置注释称“planning failure 后递归拆分（最大深度 2）”，也保留了 decomposition prompt，但当前 `generate_inner()` 在 planning 失败时直接停止，未调用递归拆分。这是文档与实际控制流的已知漂移，修改时以控制流为准。
- Stage 3 是 text-only：不向最终 reasoning 再传原图。

## V6

- 基本执行器与 V5 接近，主要变化在 Stage 3。
- 最终 reasoning 会收到原图，但提示要求只做 gross-error sanity check；视觉观察仍是主要事实来源。
- 增加“单次推理、立即提交、避免自我怀疑/反复回看”的约束，目的是减少长输出导致 JSON/答案丢失。
- 已注册为 `Qwen3.5-9B-OCRAgentV6`，默认同样 `skip_locate=True`；应视为 V5 的实验变体，而不是已经验证替代所有 V5 实验的稳定版本。

# 共同实现约定

- 所有阶段通过 `_Base64Agent` / `OpenAIWrapper` 调 OpenAI-compatible endpoint。
- 结构化输出依赖 JSON Schema/guided JSON；解析器还会尝试从 markdown fence 或混杂文本中恢复 JSON。
- bbox 使用 `[x1, y1, x2, y2]` 的千分比归一化坐标，裁剪前换算到原图像素。
- `MAX_ATTEMPTS = 3` 是 stage/step 内部尝试总数，与 wrapper 注册中的 `retry=10`（HTTP/API wrapper 重试）不是同一层重试。
- trace 写到各注册项的 `trace_dir/execution_traces.jsonl`，并含 planning、executor、reasoning、耗时和 overall status；诊断优先读 trace，再读散落日志。
- 主要失败状态包括 planning parse/API/invalid、grounding locate/understand、transcribe、compute、logical、reasoning 和 unexpected execution error。

# Step-QA 数据链

典型链路如下：

1. `step_qa_pipeline.py` 从 OCR-Reasoning 结果 Excel 解析中英文步骤，分类并生成可验证 QA；LLM 失败时可模板回退。
2. `step_qa_generate.py` 可对中间 Excel 分段补生成 QA，`verify_step_qa.py` 用于检查产物。
3. `build_step_reasoning_qa_tsv.py` 生成 `outputs/step_qa/step_reasoning_qa.tsv` 和图片引用。
4. `Step_Reasoning_QA` 在一张原图上按顺序提问多个子问题，以“全部步骤正确”作为题目级成功标准，并另算单步指标。
5. `OCR_Reasoning_StepQA` 把前 `n-1` 个 step QA 作为提示注入原题，刻意排除最后一步以防最终答案泄漏。
6. `build_depended_step_qa.py` / `build_depended_tsv.py` 构造带上游依赖提示的版本；`Depended_Step_Reasoning_QA` 和 `evaluate_depended_serial.py` 用于分析依赖传播。

这些脚本的默认输入输出路径绑定具体实验轮次，运行前必须检查文件常量或 CLI 参数，不能假设默认路径仍存在。

# 已知陷阱与优先排查顺序

- `README.md` 当前为空，这是提交历史中的显式删除结果；项目说明应先看本记忆、`CLAUDE.md` 和代码，而不是自动“恢复” README。
- `CLAUDE.md` 中把 `pgr_ocragent.py` 称为 V4 legacy，与当前 `ocr_agent_v4.py`–`v6.py` 演化线不完全一致；以注册项和实现为准。
- V5/V6 常用注册默认跳过 locate。看到 grounding 效果异常时，先确认实际模型名和 `skip_locate`，再分析 bbox。
- planning 失败不会进入 Stage 3；空答案首先查 planning trace，而不是只查 reasoning prompt。
- Step-QA 的绝对 Windows 路径只是本机候选，移植失败时先查 `outputs/step_qa/` 与 `LMU_DATA`。
- 原始 OCR-Reasoning judge 当前强制串行；“推理很快但评测卡住”可能只是 judge 单进程，不一定是死锁。
- 当前工作树常有大量实验日志删除、图片变化和未跟踪数据。任何清理/提交前都要限定文件范围。
