# OCR-Agent V4 架构设计记录 (最终版)

## 状态: ✅ 已实现

## 概述
从头重新搭建 OCR-Agent，完全基于 Qwen3.5-9B VL 模型。
API: http://10.3.112.34:22630/v1

## 三阶段流水线

### Stage 1: Planning
- 输入: 原图 (high-res, 4096×32×32 ~ 8196×32×32 via extra_body) + 问题
- 输出: JSON (guided_json constrained), steps DAG
- Schema: {steps: [{id: string, type: enum, description: string, depends_on: string[]}]}
- Step 类型: `ground-understand` | `compute` | `logical`
- depends_on 只能引用已出现的 step ID
- 失败 → fallback 到直接推理

### Stage 2: Executor (DAG-based)
- 按拓扑序顺序执行每个 step（每个 step 独立 API 调用）
- 上下文: 只传入 depends_on 指定的 step 结果
- **ground-understand** (两阶段):
  - Phase 1 (Locate): 原图 + step描述 + 依赖上下文 → bbox_groups [{group_id, purpose, bboxes}]
  - Phase 2 (Understand): 所有 bbox crops 一起输入 → {result: "..."}
  - 坐标: 千分比归一化 [x1,y1,x2,y2], pilot → pixel 转换
- **compute**: 纯文本, {result: "..."}
- **logical**: 纯文本, {result: any} 灵活输出
- 每 step 独立的 system prompt (5种: Planning/Locate/Understand/Compute/Logical)

### Stage 3: Reasoning
- 输入: 所有 steps 结果汇总 + 原图 + 问题
- 输出: CoT推理 + <a> 标签包裹的最终答案
- 无 guided decoding (free-form 推理)

## 关键技术选择
- Prompt驱动 + constrained JSON (vLLM guided_json) — 不用 function calling
- 每次独立 API 调用（无状态）
- 所有阶段共用同一 9B 模型
- base64 in-memory 子图传递（_Base64Agent）

## 错误处理 & 状态追踪
- 无重试，出错继续
- `success`: 全部成功
- `planning_failed`: Planning 失败 → fallback direct reasoning
- `grounding_error`: 任何 ground-understand step 失败
- `execution_error`: 任何 compute/logical step 失败

## 文件
- 主实现: `vlmeval/vlm/ocr_agent_v4.py` (~970 lines)
- 配置注册: `vlmeval/config.py` (Qwen3.5-9B-OCRAgentV4, Qwen3.5-27B-OCRAgentV4)
- 测试脚本: `test_ocr_agent_v4.py`

## 运行方式
```bash
# 框架模式
python run.py --data OCR_Reasoning --model Qwen3.5-9B-OCRAgentV4 --verbose

# 独立测试
python test_ocr_agent_v4.py
```
