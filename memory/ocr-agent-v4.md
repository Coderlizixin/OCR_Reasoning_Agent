---
name: ocr-agent-v4
description: OCR-Agent V4 的 Planning → Executor(DAG) → Reasoning 架构与当前实现约定
metadata:
  type: project
  last_verified: 2026-08-18
---

# OCR-Agent V4

主实现：`vlmeval/vlm/ocr_agent_v4.py`。

V4 是从旧 ThreeStage/PGR 实验线之后重新组织的单模型 agent。Planning、Locate、Understand、Transcribe、Compute、Logical 与最终 Reasoning 都通过同一 OpenAI-compatible 多模态模型服务完成，但每个阶段使用独立 system prompt 和 JSON Schema。

# 三阶段结构

## 1. Planning

输入原图与问题，输出：

- `strategy`：求解方法，不应泄露已从图中读到的具体值或最终答案。
- `steps`：包含 `id`、`type`、`description`、`depends_on` 的列表。

允许的类型是 `ground-understand`、`transcribe`、`compute`、`logical`。依赖必须指向已出现的步骤，从而保证计划天然可按拓扑顺序执行。计划 API/解析/结构失败最多尝试 3 次，并分别记录状态。

## 2. Executor

- 每步只接收其 `depends_on` 成功结果作为文本上下文。
- `ground-understand` 默认流程是 Locate → bbox 换算/裁剪 → Understand；注册项可用 `skip_locate=True` 直接向 Understand 提供整图。
- `transcribe` 用于完整转录试题、文档或区域；学科题通常先转录，再交给 compute/logical。
- `compute` 与 `logical` 不看图，只使用上游文本结果。
- 每步最多 3 次尝试。V4 对失败 visual step 还保留一次 executor 层 re-plan；后续 V5 的研究方向是移除这一机制。
- 视觉抽取可做多次调用与数字一致性/多数选择，但一致性判断主要基于提取到的数字，对纯文本差异较敏感。

bbox 为千分比归一化 `[x1, y1, x2, y2]`，转换到原图像素后裁剪。调试时不要误当作 0–1 浮点坐标。

## 3. Reasoning

汇总 strategy、image overview 与所有步骤结果：

- `ground-understand` / `transcribe` 被视为视觉观察（FACT）。
- `compute` / `logical` 被视为派生草稿（DRAFT），最终阶段应从视觉观察重新推导。
- 输出 schema 是 `{"reasoning": string, "answer": string}`，返回给 VLMEvalKit 时格式化为推理文本加 `<a>...</a>`。

planning 失败或没有可用步骤时，当前实现不会退回“原图直接作答”；这是 2026-07-16 后明确采用的 retry/fail-fast 行为。

# 注册与实验变体

注册项集中在 `vlmeval/config.py`：

- `Qwen3.5-9B-OCRAgentV4`
- `Qwen3.5-27B-OCRAgentV4`
- `Qwen3.5-9B-OCRAgentV4-PseudoPlan`
- `Qwen3.5-9B-OCRAgentV4-SkipLocate`
- `Qwen3.5-9B-OCRAgentV4-PseudoPlan-SkipLocate`

PseudoPlan 从 `test_case/planning_pseudo_labels` 读取计划以隔离 planning 误差；SkipLocate 用整图 understanding 隔离 bbox/裁剪误差。比较实验时应同时记录模型注册名、是否 pseudo plan、是否 skip locate，不能只写“V4”。

# 运行与诊断

```powershell
python run.py --data OCR_Reasoning --model Qwen3.5-9B-OCRAgentV4 --verbose
python test_ocr_agent_v4.py
python test_ocr_agent_v4_pseudo_plan.py
python test_ocr_agent_v4_skip_locate.py
```

服务地址、模型别名、图像 processor 参数与 trace 目录以 `vlmeval/config.py` 当前注册项为准，不在记忆中固化。单例排障依次查看：planning JSON → step dependency/context → locate bbox → crop/understand → final trace/status → judge 结果。
