---
name: ocr-agent-v4
description: OCR-Agent V4 architecture — Planning→Executor(DAG)→Reasoning pipeline using single 9B model
metadata:
  type: project
---

OCR-Agent V4 is a from-scratch rebuild of the OCR agent, using only Qwen3.5-9B VL model.

**Architecture**: Three-stage pipeline
- Stage 1 (Planning): image+question → JSON DAG of steps (ground-understand/compute/logical) with dependencies
- Stage 2 (Executor): topological execution, each step independent API call with constrained JSON output. ground-understand has Locate→Crop→Understand phases
- Stage 3 (Reasoning): all step results + original image + question → CoT + final answer

**Key Design Decisions**:
- Prompt-driven JSON output with vLLM guided_json (not function calling)
- DAG-based context: each step only sees dependent steps' results
- No retry on errors; final status tracked (success/planning_failed/grounding_error/execution_error)
- Planning failure → fallback to direct image+question reasoning
- Bbox coordinates: per-mille normalized [x1,y1,x2,y2], converted to pixels for cropping
- All 3 step types have independent system prompts (Planning/Locate/Understand/Compute/Logical/Reasoning)

**Why**: Previous V2 and V3 attempts had poor results. V4 simplifies to a clean DAG-based architecture.

**How to apply**: 
- Main implementation: `vlmeval/vlm/ocr_agent_v4.py`
- Registered in `vlmeval/config.py` as `Qwen3.5-9B-OCRAgentV4` and `Qwen3.5-27B-OCRAgentV4`
- Run with: `python run.py --data OCR_Reasoning --model Qwen3.5-9B-OCRAgentV4 --verbose`
- Test script: `test_ocr_agent_v4.py`
- API endpoint: http://10.3.112.34:22630/v1 (model: Qwen3.5-9B)
