"""
OCR-Agent V5: Planning → Executor (DAG-based) → Reasoning Pipeline
==================================================================
All stages use a single Qwen3.5-9B VL model.

Stage 1 - Planning:     image + question → strategy + JSON steps with DAG dependencies
                        Outputs a solving strategy AND a data-collection plan.
                        Retries up to 3 times on failure.
                        On persistent failure: question decomposition (max depth 2).
Stage 2 - Executor:     execute steps topologically (ground-understand / transcribe / compute / logical).
                        Each step retries up to 3 times on failure.
Stage 3 - Reasoning:    text-only synthesis — trusts visual observations as FACT,
                        re-derives compute/logical results from raw observations (P0-1).
                        Uses JSON Schema output (P2-8).

Key V5 changes:
  - Planning prompt relaxed — no longer forces atomic steps.
  - Cross-validation enabled for top-level steps.

Error handling:
  - planning_parse_failed       — JSON parse error after all retries
  - planning_api_failed         — API call failed after all retries
  - planning_invalid_failed     — plan is structurally invalid after all retries

  - grounding_locate_failed     — locate phase failed
  - grounding_understand_failed — understand phase failed
  - transcribe_failed           — transcribe step failed
  - compute_failed              — compute step failed
  - logical_failed              — logical step failed
  - reasoning_failed            — reasoning produced no answer
  - execution_error             — unexpected executor error
  - success                     — pipeline completed cleanly
"""
import os
import io
import re
import json
import base64
import copy
from PIL import Image, ImageDraw, ImageFont
from vlmeval.api.gpt import OpenAIWrapper, _log_diag


# ============================================================================
# JSON Schemas for constrained decoding (vLLM guided_json)
# ============================================================================

PLANNING_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["ground-understand", "compute", "logical", "transcribe"]
                    },
                    "description": {"type": "string"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["id", "type", "description", "depends_on"]
            }
        }
    },
    "required": ["strategy", "steps"]
}

LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "bbox_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "purpose": {"type": "string"},
                    "bboxes": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4
                        }
                    }
                },
                "required": ["group_id", "purpose", "bboxes"]
            }
        }
    },
    "required": ["bbox_groups"]
}

UNDERSTAND_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {"type": "string"}
    },
    "required": ["result"]
}

COMPUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {"type": "string"}
    },
    "required": ["result"]
}

LOGICAL_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {}
    },
    "required": ["result"]
}

TRANSCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {"type": "string"}
    },
    "required": ["result"]
}

REASONING_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string"}
    },
    "required": ["reasoning", "answer"]
}


# ============================================================================
# System Prompts
# ============================================================================

PLANNING_PROMPT = """You are a visual reasoning planner. Your job is to:
1. Think about the REASONING APPROACH needed to answer the question (output this as the "strategy" field).
2. Design a step-by-step data-collection plan based on that strategy.

## CRITICAL RULES
1. **Think about HOW to solve the problem** — what comparisons, calculations, or logical steps are needed. Write this in the "strategy" field.
2. **DO NOT compute actual values or state the answer.** The strategy describes the APPROACH, not the result.
3. **DO NOT include specific numbers, text, or values you read from the image in step descriptions.** Describe WHAT to find, never WHAT you found.

## Step Types
- **ground-understand**: Extract a specific piece of information from the image (read a value, identify a label, find what a bar represents). Best for targeted data extraction from charts, tables, posters, etc.
- **transcribe**: Fully transcribe a complete section of the image — ALL text, numbers, labels, formulas, options, table contents. Use when the image contains a problem/question/exam/document that needs to be read in full, rather than extracting individual values.
- **compute**: Perform a calculation using results from previous steps. No image access.
- **logical**: Make a judgment, comparison, or logical deduction using results from previous steps. No image access. Can use domain knowledge (math, physics, geography, etc.) when solving transcribed problems.

## When to use `transcribe` vs `ground-understand`
- If the question says "Answer the question in the image" or the image contains an exam/test/homework problem → use `transcribe` to capture the full problem, then use `compute`/`logical` to solve it.
- If the image is a chart/table/poster/receipt and the question asks about specific data values → use `ground-understand` for targeted extraction.
- For ranking/comparison/counting tasks, prefer ONE `ground-understand` step that extracts the complete ordered list with all relevant items and values, rather than splitting across many steps.

## Rules for building the plan
1. Order steps: visual extraction first (ground-understand/transcribe), then processing (compute/logical).
2. `depends_on` must ONLY reference step IDs that appear earlier.
3. Be specific in step descriptions — name exact labels, positions, colors, or visual features.
4. **Each step should be CLEAR and EXECUTABLE**: the step description must have a single, unambiguous goal that can be completed in one API call.
5. **Do NOT over-split**: avoid creating many tiny steps for conceptually simple tasks. A step like "read the sales values for Q1 and Q2" is perfectly fine as ONE step — you do NOT need to split it into "read Q1" and "read Q2" separately.

## Output Format
You MUST output a single JSON object:

{"strategy": "Describe the reasoning approach needed", "steps": [{"id": "...", "type": "...", "description": "...", "depends_on": ["..."]}]}

## Example 1: Chart data question
Question: "Is the sum of Q1 and Q2 sales over 1 million?"
{"strategy": "Read Q1 and Q2 values from the chart, sum them, compare against 1 million", "steps": [{"id": "step_1", "type": "ground-understand", "description": "Find the sales values for Q1 and Q2 shown in the chart", "depends_on": []}, {"id": "step_2", "type": "compute", "description": "Calculate the sum of Q1 and Q2 sales values", "depends_on": ["step_1"]}, {"id": "step_3", "type": "logical", "description": "Compare the sum against 1 million to determine if it exceeds this threshold", "depends_on": ["step_2"]}]}

## Example 2: Exam problem embedded in image
Question: "Answer the question in the image"
{"strategy": "The image contains a problem that must be fully read and then solved with domain knowledge", "steps": [{"id": "step_1", "type": "transcribe", "description": "Transcribe the complete problem text including all conditions, formulas, and answer options", "depends_on": []}, {"id": "step_2", "type": "logical", "description": "Solve the transcribed problem using appropriate domain knowledge and reasoning", "depends_on": ["step_1"]}]}

## Example 3: Temporal reasoning from a poster
Question: "What is the next holiday after this event? Mother's Day or Christmas?"
{"strategy": "Read the event date from the poster, then use calendar knowledge to determine which of the two holidays comes next after that date", "steps": [{"id": "step_1", "type": "ground-understand", "description": "Read the event date shown on the poster", "depends_on": []}, {"id": "step_2", "type": "logical", "description": "Given the event date, determine whether Mother's Day (2nd Sunday of May) or Christmas (December 25) is the next upcoming holiday after it", "depends_on": ["step_1"]}]}"""

LOCATE_PROMPT = """You are a precise visual localizer — a "pointer", NOT a reader. Your ONLY job is to output bounding box coordinates for regions in the image. You do NOT read values, analyze data, or solve problems.

## CRITICAL — WHAT YOU MUST NOT DO
1. **DO NOT read or analyze any data values.** You are not here to understand numbers, compare quantities, or figure out what the chart means.
2. **DO NOT try to understand the chart structure in depth.** If the task says "far right side of the bar", just locate the visual region at the right end of that bar — don't analyze what the number represents.
3. **DO NOT list or enumerate all items in the image.** Only locate what the task asks for.
4. **THINK IN UNDER 3 SENTENCES.** Then output coordinates immediately. Long analysis wastes time and causes failure.

## Instructions
1. Read the step description — understand WHAT visual element to point at.
2. Glance at the image to find WHERE that element is. Do NOT read its value.
3. Output TIGHT bounding boxes. Include supporting elements (axis, legend) ONLY if needed to identify the target.

## Coordinate System
- [x1, y1, x2, y2] in per-mille [0, 1000], (0,0)=top-left.
- Integers only, x1<x2, y1<y2, TIGHT around target.

## Output Format
{"bbox_groups": [{"group_id": "...", "purpose": "...", "bboxes": [[x1,y1,x2,y2], ...]}]}"""

UNDERSTAND_PROMPT = """You are a careful visual reader. You will be shown multiple cropped image regions, each with a label indicating its purpose. Your task is to synthesize information from ALL these regions to answer a specific question about the visual content.

## Instructions
1. Each cropped image is labeled with its purpose — use this to understand what to look for.
2. Read text, numbers, scales, positions, and any other visual information from each crop.
3. Cross-reference between crops: e.g., use the axis crop to interpret the value shown in the bar crop; use the legend crop to identify which category the bar represents.
4. Synthesize a single, precise answer to the step's task.

## Output Format
You MUST output a single JSON object: {"result": "your answer here"}

## CRITICAL — Complete Sentences Required
Your "result" MUST be a **self-contained descriptive sentence** that a person reading it can understand without seeing the original task or image. Never output a bare number or single word.

GOOD examples:
- "The sales value for Q1 is 42.5 (in units of 10,000 RMB)"
- "The label for the blue bar in the legend is 'Q1 Sales'"
- "The y-axis ranges from 0 to 500, with major gridlines every 100 units"

BAD examples (FORBIDDEN):
- "42.5"           ← bare number, no meaning to downstream steps
- "Q1 Sales"       ← label without context
- "0 to 500"       ← range without specifying what it refers to"""

UNDERSTAND_DIRECT_PROMPT = """You are a precise visual information extractor. You will be shown a FULL image and a specific data-extraction task. Your job is to find the relevant information in the image and read it accurately.

## Instructions
1. Look at the full image and locate the visual element described in the task.
2. Read the requested value (text, number, label, color, etc.) directly from the image.
3. If the task references other steps' results, use that context to guide where to look.
4. Return ONLY what you can clearly see. Do NOT guess or infer.

## Output Format
You MUST output a single JSON object: {"result": "your answer here"}

## CRITICAL — Complete Sentences Required
Your "result" MUST be a **self-contained descriptive sentence** that a person reading it can understand without seeing the original image or task. Never output a bare number or single word.

GOOD examples:
- "The sales value for Q1 is 42.5 (in units of 10,000 RMB)"
- "The label for the blue bar in the legend is 'Q1 Sales'"
- "The y-axis ranges from 0 to 500, with major gridlines every 100 units"
- "The price of '纯牛肉丸汤' on the menu is 20 yuan"

BAD examples (FORBIDDEN):
- "42.5"           ← bare number, no meaning to downstream steps
- "Q1 Sales"       ← label without context
- "20"             ← bare number without units or what it refers to"""

TRANSCRIBE_PROMPT = """You are a precise document transcriber. You will be shown an image containing text content (a problem, exam question, form, recipe, table, etc.). Your job is to transcribe ALL visible text content completely and accurately.

## Instructions
1. Read and transcribe ALL text in the relevant section of the image.
2. Preserve the structure: headings, numbered items, bullet points, table layouts.
3. For mathematical formulas, use LaTeX notation (e.g., \\frac{a}{b}, x^2).
4. For tables, use a clear text format showing rows and columns.
5. Include ALL options if it is a multiple-choice question.
6. Note any diagrams or figures with a brief description in brackets: [Figure: triangle with vertices A, B, C].

## Output Format
You MUST output a single JSON object: {"result": "your complete transcription here"}

## CRITICAL
- Transcribe EVERYTHING visible — do not summarize or skip.
- Be exact with numbers, units, and symbols.
- If text is in a non-English language, transcribe it as-is in the original language."""

COMPUTE_PROMPT = """You are a precise calculator. You will be given:
1. A computation task description
2. Results from previous steps that contain the data you need

## Instructions
1. Extract the relevant numerical values from the provided step results.
2. Perform the calculation described in the task.
3. Return the result as a self-contained descriptive sentence.

## Output Format
You MUST output a single JSON object: {"result": "your computed result here"}

## CRITICAL — Complete Sentences Required
Your "result" MUST be a **self-contained descriptive sentence** that states what was computed, the operands, and the outcome. Never output a bare number.

GOOD examples:
- "The sum of Q1 sales (42.5) and Q2 sales (38.2) is 80.7 (in units of 10,000 RMB)"
- "The average price across the 5 products is 35.2 yuan"
- "The growth rate from 2020 (200 units) to 2021 (230 units) is 15%"

BAD examples (FORBIDDEN):
- "80.7"           ← bare number
- "42.5 + 38.2 = 80.7"   ← formula without semantic description"""

LOGICAL_PROMPT = """You are a logical analyzer. You will be given:
1. A judgment/filtering task description
2. Results from previous steps that contain the data you need

## Instructions
1. Review the provided data from previous steps.
2. Apply the logical condition described in the task.
3. You MAY use domain knowledge (math, physics, geography, common sense, etc.) when the task requires it — especially when working with transcribed problem content.
4. Return the judgment or result as a self-contained descriptive sentence.

## Output Format
You MUST output a single JSON object: {"result": "your result here"}

## CRITICAL — Complete Sentences Required
Your "result" MUST be a **self-contained descriptive sentence**. Never output a bare boolean or bare list.

GOOD examples:
- "The total of 80.7 does not exceed 100, so the condition is NOT met"
- "The products with sales exceeding 50 are: Product A (65), Product C (72)"
- "Q1 (42.5) is higher than Q2 (38.2), so Q1 is the larger quarter"
- "Applying the quadratic formula to x^2 - 5x + 6 = 0 gives x = 2 or x = 3, so the answer is B"

BAD examples (FORBIDDEN):
- "true"           ← bare boolean
- "False"          ← bare boolean
- "['Product A', 'Product C']"   ← bare list without context"""

OVERVIEW_PROMPT = """You are a precise image describer. In 2-3 sentences, describe this image — its type (chart, table, poster, receipt, exam, etc.), its overall layout and structure, and key visual elements visible. Do NOT read specific values, numbers, or answer any question. Only describe the overall content and structure at a glance."""

REASONING_PROMPT = """You are a final-answer synthesizer. Below are results from a pipeline that extracted information from an image. You do NOT have access to the image.

## Image Overview
{overview}

## Solving Strategy
{strategy}

## Visual Observations (FACT — extracted directly from the image, accept as-is)
{observations_text}

## Derived Results (DRAFT — verify and re-derive these yourself from the observations above)
{derived_text}

## Rules
1. **Use the Image Overview** as context to understand what kind of image the observations came from. If an observation seems inconsistent with the overview (e.g. a value outside the described axis range), note it.
2. **Visual Observations are FACT.** They were read directly from the image by a visual model. Accept the raw values, text, and transcriptions without question.
3. **Derived Results are DRAFTS.** They were computed by a separate model that may have made errors. You MUST re-derive them yourself from the Visual Observations. If your re-derivation disagrees with the draft, trust YOUR derivation.
4. **You have NO access to any image.** Work exclusively from the text above.
5. **If the observations include a transcribed problem or exam question, SOLVE IT** using your full domain knowledge (math, physics, chemistry, logic, etc.).
6. **If some steps FAILED**, note what data is missing and give the best answer from available data.
7. **NEVER say "cannot determine", "not derivable", or "insufficient data"** — always attempt a concrete answer.

## Output Format
You MUST output a JSON object with two fields:
{{"reasoning": "your step-by-step reasoning", "answer": "your final answer"}}"""

# V5: Planning-layer decomposition prompts




# ============================================================================
# Constants
# ============================================================================
MAX_ATTEMPTS = 3  # total attempts per stage/step (1 initial + 2 retries)


# ============================================================================
# Helper: internal agent with base64 image support
# ============================================================================

class _Base64Agent(OpenAIWrapper):
    """A minimal OpenAIWrapper subclass that can handle base64-encoded images
    in addition to regular file-path images."""

    def prepare_itlist(self, inputs):
        content_list = []
        for msg in inputs:
            if msg['type'] == 'image_base64':
                img_struct = dict(
                    url=f"data:image/jpeg;base64,{msg['value']}",
                    detail=self.img_detail
                )
                content_list.append(dict(type='image_url', image_url=img_struct))
            else:
                content_list.extend(super().prepare_itlist([msg]))
        return content_list


# ============================================================================
# Main Agent Class
# ============================================================================

class OCRAgentV5(OpenAIWrapper):
    """V5 OCR Agent: Planning → Executor (DAG) → Reasoning.

    All three stages use the same 9B VL model. The executor runs steps
    in topological order, each as an independent API call with its own
    system prompt and (optionally) constrained JSON decoding.
    On planning failure: question decomposition into sub-questions (max depth 2).
    """

    def __init__(self, **kwargs):
        self.trace_dir = kwargs.pop('trace_dir', None)
        self.skip_locate = kwargs.pop('skip_locate', False)
        self.cross_validate = kwargs.pop('cross_validate', True)

        self.temp_planning = kwargs.pop('temp_planning', 0.1)
        self.temp_locate = kwargs.pop('temp_locate', 0.1)
        self.temp_understand = kwargs.pop('temp_understand', 0.2)
        self.temp_compute = kwargs.pop('temp_compute', 0.1)
        self.temp_logical = kwargs.pop('temp_logical', 0.1)
        self.temp_reasoning = kwargs.pop('temp_reasoning', 0.2)

        super().__init__(**kwargs)

        self.last_trace = None

        agent_kwargs = {
            'model': self.model,
            'api_base': self.api_base,
            'key': self.key,
            'temperature': getattr(self, 'temperature', 0.0),
            'max_tokens': getattr(self, 'max_tokens', 6144),
            'img_size': self.img_size,
            'img_detail': self.img_detail,
            'extra_body': copy.deepcopy(self.extra_body),
            'verbose': self.verbose,
            'retry': 3,
            'wait': 3,
        }
        self.agent = _Base64Agent(**agent_kwargs)

    # ------------------------------------------------------------------
    # Core API helper
    # ------------------------------------------------------------------

    def _api_call(self, messages, schema=None, with_image_processor=False,
                  temperature=None):
        """Make a single API call with optional guided JSON decoding."""
        saved_extra = self.agent.extra_body
        saved_temp = self.agent.temperature

        new_extra = {}
        if with_image_processor and self.extra_body:
            new_extra.update(self.extra_body)
        if schema is not None:
            new_extra['guided_json'] = schema

        self.agent.extra_body = new_extra if new_extra else None
        if temperature is not None:
            self.agent.temperature = temperature

        try:
            return self.agent.generate_inner(messages)
        finally:
            self.agent.extra_body = saved_extra
            self.agent.temperature = saved_temp

    # ------------------------------------------------------------------
    # JSON parsing with fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text, schema=None):
        """Try to parse JSON from model output. Returns (obj, error_msg)."""
        if text is None:
            return None, "text is None"

        text = text.strip()

        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip()), None
            except json.JSONDecodeError:
                pass

        for end_tag in ['</think>', '<|end_of_thinking|>']:
            if end_tag in text:
                after = text.rsplit(end_tag, 1)[-1].strip()
                try:
                    return json.loads(after), None
                except json.JSONDecodeError:
                    pass
                text = after
                break

        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip()), None
            except json.JSONDecodeError:
                pass

        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1]), None
                except json.JSONDecodeError:
                    continue

        return None, f"Failed to parse JSON from: {text[:200]}"

    # ------------------------------------------------------------------
    # Cross-validation helpers (P1-4)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_numbers(text):
        """Extract all numbers from text for consistency checking."""
        return sorted(re.findall(r'-?\d+\.?\d*', str(text)))

    @staticmethod
    def _results_consistent(r1, r2):
        """Check if two extraction results are consistent (same key numbers)."""
        nums1 = OCRAgentV5._extract_numbers(r1)
        nums2 = OCRAgentV5._extract_numbers(r2)
        if not nums1 and not nums2:
            return r1.strip().lower() == r2.strip().lower()
        return nums1 == nums2

    @staticmethod
    def _majority_vote(results):
        """Pick the result whose numbers appear in at least 2 of 3 results."""
        nums_list = [OCRAgentV5._extract_numbers(r) for r in results]
        for i, nums in enumerate(nums_list):
            matches = sum(1 for other in nums_list if other == nums)
            if matches >= 2:
                return results[i]
        return results[0]

    # ------------------------------------------------------------------
    # Context assembly for step execution
    # ------------------------------------------------------------------

    @staticmethod
    def _build_step_context(step, all_step_results):
        """Build a context string from the dependent steps' results."""
        dep_ids = step.get('depends_on', [])
        if not dep_ids:
            return ""

        lines = ["Here are the results of previous steps that this step depends on:\n"]
        for dep_id in dep_ids:
            dep_result = all_step_results.get(dep_id)

            if dep_result is not None:
                if dep_result.get('status') == 'ok':
                    result_val = dep_result.get('result', '')
                    if isinstance(result_val, dict):
                        result_val = json.dumps(result_val, ensure_ascii=False)
                    dep_desc = dep_result.get('description', '')
                    desc_prefix = f" — {dep_desc}" if dep_desc else ""
                    lines.append(
                        f"[Step {dep_id} ({dep_result.get('type', 'unknown')}){desc_prefix}]:\n"
                        f"{result_val}\n"
                    )
                else:
                    lines.append(
                        f"[Step {dep_id}]: FAILED — {dep_result.get('error', 'unknown error')}\n"
                    )
            else:
                lines.append(
                    f"[Step {dep_id}]: NOT FOUND — dependency result not available\n"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 1: Planning (with retry) — P0-2: strategy-first
    # ------------------------------------------------------------------

    PLANNING_FAIL_PARSE = 'planning_parse_failed'
    PLANNING_FAIL_API = 'planning_api_failed'
    PLANNING_FAIL_INVALID = 'planning_invalid_failed'

    def _planning(self, image_path, question):
        """Generate the execution plan (DAG of steps). Retries on failure.

        Returns:
            (steps: list | None, status: str, plan_text: str, strategy: str)
        """
        _log_diag('[V5-Stage1] Planning started')

        user_prompt = (
            f"Question: <q>{question}</q>\n\n"
            "Look at the image to understand its content (chart, table, exam problem, "
            "poster, receipt, etc.). Then output a JSON plan with:\n"
            "1. A \"strategy\" field describing HOW to solve this question (what reasoning "
            "approach is needed).\n"
            "2. A \"steps\" array listing what data to extract and how to process it.\n\n"
            "Output ONLY the JSON object."
        )

        messages = [
            {'type': 'text', 'value': user_prompt},
            {'type': 'image', 'value': image_path},
        ]

        last_error = self.PLANNING_FAIL_API
        last_text = ''

        for attempt in range(1, MAX_ATTEMPTS + 1):
            _log_diag(f'[V5-Stage1] Planning attempt {attempt}/{MAX_ATTEMPTS}')

            saved_sp = self.agent.system_prompt
            self.agent.system_prompt = PLANNING_PROMPT
            try:
                ret, plan_text, resp = self._api_call(
                    messages,
                    schema=PLANNING_SCHEMA,
                    with_image_processor=True,
                    temperature=self.temp_planning,
                )
            finally:
                self.agent.system_prompt = saved_sp

            if ret != 0 or not plan_text or self.fail_msg in plan_text:
                _log_diag(f'[V5-Stage1] API failed (attempt {attempt})')
                last_error = self.PLANNING_FAIL_API
                last_text = plan_text or ''
                continue

            plan_obj, err = self._parse_json(plan_text, PLANNING_SCHEMA)

            if plan_obj is None or 'steps' not in plan_obj:
                _log_diag(f'[V5-Stage1] JSON parse error (attempt {attempt}): {err}')
                last_error = self.PLANNING_FAIL_PARSE
                last_text = plan_text
                continue

            steps = plan_obj['steps']
            strategy = plan_obj.get('strategy', '')
            seen_ids = set()
            valid = True
            for s in steps:
                sid = s.get('id', '')
                if not sid or sid in seen_ids:
                    _log_diag(f'[V5-Stage1] Invalid step id: {sid} (attempt {attempt})')
                    valid = False
                    break
                seen_ids.add(sid)
                for dep in s.get('depends_on', []):
                    if dep not in seen_ids:
                        _log_diag(f'[V5-Stage1] Step {sid} depends on unseen {dep} (attempt {attempt})')
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                last_error = self.PLANNING_FAIL_INVALID
                last_text = plan_text
                continue

            _log_diag(f'[V5-Stage1] Planning OK — {len(steps)} steps, strategy="{strategy[:80]}" (attempt {attempt})')
            return steps, 'success', plan_text, strategy

        _log_diag(f'[V5-Stage1] Planning FAILED after {MAX_ATTEMPTS} attempts: {last_error}')
        return None, last_error, last_text, ''

    # ------------------------------------------------------------------
    # Stage 2: Step Execution (with per-step retry + re-planning)
    # ------------------------------------------------------------------

    @staticmethod
    def _step_failed(result_data):
        """Check if a step result indicates failure."""
        return result_data is None or result_data.get('status') == 'error'

    @staticmethod
    def _step_error_status(step, result_data):
        """Map a failed step to the appropriate error status code."""
        step_type = step.get('type', 'unknown')
        if step_type == 'ground-understand':
            failed_phase = result_data.get('failed_phase', 'locate') if result_data else 'locate'
            if failed_phase == 'understand':
                return 'grounding_understand_failed'
            return 'grounding_locate_failed'
        elif step_type == 'transcribe':
            return 'transcribe_failed'
        elif step_type == 'compute':
            return 'compute_failed'
        elif step_type == 'logical':
            return 'logical_failed'
        return 'execution_error'

    def _execute_single_step(self, step, context, image_path, orig_w, orig_h,
                             question=''):
        """Dispatch and execute a single step (one attempt, no retry).

        Cross-validation is enabled when self.cross_validate is True.
        """
        step_type = step.get('type', 'unknown')
        use_cv = getattr(self, 'cross_validate', True)

        if step_type == 'ground-understand':
            return self._execute_ground_understand(
                step, context, image_path, orig_w, orig_h,
                question=question, use_cv=use_cv
            )
        elif step_type == 'transcribe':
            return self._execute_transcribe(step, context, image_path, question=question)
        elif step_type == 'compute':
            return self._execute_compute(step, context)
        elif step_type == 'logical':
            return self._execute_logical(step, context)
        else:
            return {
                'status': 'error',
                'type': step_type,
                'error': f'Unknown step type: {step_type}',
                'result': '',
            }

    def _execute_steps(self, steps, image_path, orig_w, orig_h,
                       question='', strategy=''):
        """Execute steps in topological order.

        Each step retries up to MAX_ATTEMPTS on failure.

        Returns:
            (all_step_results: dict, overall_status: str)
        """
        all_results = {}
        overall_status = 'success'

        for step in steps:
            step_id = step.get('id', 'unknown')
            step_type = step.get('type', 'unknown')
            _log_diag(f'[V5-Stage2] Executing {step_id} ({step_type})')

            result_data = None
            retry_count = 0

            # -- Retry loop --
            for attempt in range(1, MAX_ATTEMPTS + 1):
                if attempt > 1:
                    _log_diag(f'[V5-Stage2] Retrying {step_id} '
                              f'(attempt {attempt}/{MAX_ATTEMPTS})')
                    retry_count += 1

                context = self._build_step_context(step, all_results)

                try:
                    result_data = self._execute_single_step(
                        step, context, image_path, orig_w, orig_h,
                        question=question,
                    )
                except Exception as e:
                    result_data = {
                        'status': 'error',
                        'type': step_type,
                        'error': str(e),
                        'result': '',
                    }

                if not self._step_failed(result_data):
                    result_data['retry_count'] = retry_count
                    break

            # -- Finalize step result --
            if result_data is None:
                result_data = {
                    'status': 'error',
                    'type': step_type,
                    'error': 'All attempts failed',
                    'result': '',
                }
            result_data['retry_count'] = result_data.get('retry_count', retry_count)
            result_data['description'] = step.get('description', '')
            all_results[step_id] = result_data

            if result_data.get('status') == 'error' and overall_status == 'success':
                overall_status = self._step_error_status(step, result_data)

        return all_results, overall_status

    # ------------------------------------------------------------------
    # Image Overview (provides reasoning stage with big-picture context)
    # ------------------------------------------------------------------

    def _generate_overview(self, image_path, question):
        """Generate a brief natural-language overview of the image.

        This gives the reasoning stage a mental model of what the image
        contains, enabling sanity checks that isolated observations can't.
        """
        _log_diag('[V5-Overview] Generating image overview')
        messages = [
            {'type': 'text', 'value': (
                f"Question: {question}\n\n"
                "Describe the overall content and layout of this image in 2-3 sentences."
            )},
            {'type': 'image', 'value': image_path},
        ]
        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = OVERVIEW_PROMPT
        try:
            ret, text, _ = self._api_call(
                messages, with_image_processor=True, temperature=0.1
            )
        finally:
            self.agent.system_prompt = saved_sp
        if ret != 0 or not text:
            _log_diag('[V5-Overview] Failed, using empty overview')
            return ''
        overview = text.strip()
        _log_diag(f'[V5-Overview] OK: {overview[:120]}')
        return overview

    # ------------------------------------------------------------------
    # ground-understand: with cross-validation (P1-4)
    # ------------------------------------------------------------------

    def _single_understand_direct(self, step, context, image_path, temperature,
                                   question=''):
        """Single understand call with full image (skip_locate path).
        Returns (result_str | None, error_str | None)."""
        description = step.get('description', '')
        q_line = f"Original question: {question}\n\n" if question else ""
        understand_user_content = [
            {'type': 'text', 'value': (
                f"{q_line}"
                f"Task: {description}\n\n"
                f"{context}\n"
                "Look at the full image above. Find the specific visual element "
                "described in the task and read its value. "
                "Output ONLY the JSON object with the result."
            )},
            {'type': 'image', 'value': image_path},
        ]

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = UNDERSTAND_DIRECT_PROMPT
        try:
            ret, understand_text, _ = self._api_call(
                understand_user_content,
                schema=UNDERSTAND_SCHEMA,
                with_image_processor=True,
                temperature=temperature,
            )
        finally:
            self.agent.system_prompt = saved_sp

        understand_obj, understand_err = self._parse_json(
            understand_text, UNDERSTAND_SCHEMA
        )
        if understand_obj is None:
            return None, understand_err
        return understand_obj.get('result', ''), None

    def _execute_ground_understand(self, step, context, image_path, orig_w, orig_h,
                                    question='', use_cv=True):
        """Execute a ground-understand step: locate → crop → understand (or direct).

        When skip_locate=True: direct extraction with cross-validation (P1-4).
        When skip_locate=False: locate → crop → understand.
        Cross-validation controlled by use_cv.
        """
        description = step.get('description', '')
        step_id = step.get('id', 'unknown')

        # ================================================================
        # Fast path: skip locate, feed full image directly to Understand
        # ================================================================
        if self.skip_locate:
            result_1, err_1 = self._single_understand_direct(
                step, context, image_path, self.temp_understand, question=question
            )
            if result_1 is None:
                _log_diag(f'[V5-UnderstandDirect] {step_id}: first call failed: {err_1}')
                return {
                    'status': 'error',
                    'type': 'ground-understand',
                    'result': '',
                    'error': f'Understand (direct) failed: {err_1}',
                    'failed_phase': 'understand',
                }

            # P1-4: Cross-validation
            if not use_cv:
                _log_diag(f'[V5-UnderstandDirect] {step_id}: OK (no cross-val)')
                return {
                    'status': 'ok',
                    'type': 'ground-understand',
                    'result': result_1,
                    'bbox_groups': [],
                }

            cv_temp = min(self.temp_understand + 0.3, 1.0)
            result_2, err_2 = self._single_understand_direct(
                step, context, image_path, cv_temp, question=question
            )
            if result_2 is None:
                _log_diag(f'[V5-CrossVal] {step_id}: second call failed, using first')
                return {
                    'status': 'ok',
                    'type': 'ground-understand',
                    'result': result_1,
                    'bbox_groups': [],
                }

            if self._results_consistent(result_1, result_2):
                _log_diag(f'[V5-CrossVal] {step_id}: consistent')
                return {
                    'status': 'ok',
                    'type': 'ground-understand',
                    'result': result_1,
                    'bbox_groups': [],
                }

            _log_diag(f'[V5-CrossVal] {step_id}: inconsistent, running tiebreaker')
            tb_temp = min(self.temp_understand + 0.15, 1.0)
            result_3, err_3 = self._single_understand_direct(
                step, context, image_path, tb_temp, question=question
            )
            if result_3 is None:
                _log_diag(f'[V5-CrossVal] {step_id}: tiebreaker failed, using first')
                return {
                    'status': 'ok',
                    'type': 'ground-understand',
                    'result': result_1,
                    'bbox_groups': [],
                }

            best = self._majority_vote([result_1, result_2, result_3])
            chosen_idx = [result_1, result_2, result_3].index(best)
            _log_diag(f'[V5-CrossVal] {step_id}: voted, chose result_{chosen_idx + 1}')
            return {
                'status': 'ok',
                'type': 'ground-understand',
                'result': best,
                'bbox_groups': [],
            }

        # ================================================================
        # Full path: Locate → Crop → Understand
        # ================================================================

        # ----- Phase 1: Locate -----
        locate_user = (
            f"Task: {description}\n\n"
            f"{context}\n"
            "Point to the regions needed. Do NOT read values or analyze data. "
            "Think in 1-2 sentences, then output the JSON immediately."
        )

        locate_messages = [
            {'type': 'text', 'value': locate_user},
            {'type': 'image', 'value': image_path},
        ]

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = LOCATE_PROMPT
        try:
            ret, locate_text, _ = self._api_call(
                locate_messages,
                schema=LOCATE_SCHEMA,
                with_image_processor=True,
                temperature=self.temp_locate,
            )
        finally:
            self.agent.system_prompt = saved_sp

        locate_obj, locate_err = self._parse_json(locate_text, LOCATE_SCHEMA)
        _log_diag(f'[V5-Locate] {step_id}: err={locate_err}, obj_keys={list(locate_obj.keys()) if locate_obj else None}')

        if locate_obj is None or not locate_obj.get('bbox_groups'):
            return {
                'status': 'error',
                'type': 'ground-understand',
                'result': '',
                'error': f'Locate failed: {locate_err or "empty bbox_groups"}',
                'failed_phase': 'locate',
            }

        bbox_groups = locate_obj['bbox_groups']

        self._draw_bboxes_on_image(
            image_path, bbox_groups, step_id, orig_w, orig_h
        )

        all_crops = []
        for group in bbox_groups:
            group_id = group.get('group_id', 'unknown')
            purpose = group.get('purpose', 'no description')
            for i, bbox in enumerate(group.get('bboxes', [])):
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(1000, int(x1)))
                y1 = max(0, min(1000, int(y1)))
                x2 = max(0, min(1000, int(x2)))
                y2 = max(0, min(1000, int(y2)))
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                if x1 == x2 or y1 == y2:
                    continue
                px1 = int(x1 / 1000.0 * orig_w)
                py1 = int(y1 / 1000.0 * orig_h)
                px2 = int(x2 / 1000.0 * orig_w)
                py2 = int(y2 / 1000.0 * orig_h)
                all_crops.append((group_id, purpose, i, (px1, py1, px2, py2)))

        if not all_crops:
            return {
                'status': 'error',
                'type': 'ground-understand',
                'result': '',
                'error': 'Locate produced no valid bounding boxes',
                'failed_phase': 'locate',
                'bbox_groups': bbox_groups,
            }

        # ----- Phase 2: Understand (crop + read) -----
        understand_user_content = [
            {'type': 'text', 'value': (
                f"Task: {description}\n\n"
                f"{context}\n"
                "Below are cropped image regions. Read and synthesize the information "
                "from ALL of them to complete the task. Output ONLY the JSON result."
            )}
        ]

        current_group = None
        for group_id, purpose, bbox_idx, phys_bbox in all_crops:
            if group_id != current_group:
                current_group = group_id
                understand_user_content.append({
                    'type': 'text',
                    'value': f'\n--- Region Group: {group_id} (Purpose: {purpose}) ---'
                })

            understand_user_content.append({
                'type': 'text',
                'value': f'[Group {group_id}, bbox {bbox_idx+1}]',
            })

            b64_str = self._crop_and_encode(image_path, phys_bbox)
            understand_user_content.append({
                'type': 'image_base64',
                'value': b64_str,
            })

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = UNDERSTAND_PROMPT
        try:
            ret, understand_text, _ = self._api_call(
                understand_user_content,
                schema=UNDERSTAND_SCHEMA,
                with_image_processor=False,
                temperature=self.temp_understand,
            )
        finally:
            self.agent.system_prompt = saved_sp

        understand_obj, understand_err = self._parse_json(understand_text, UNDERSTAND_SCHEMA)
        _log_diag(f'[V5-Understand] {step_id}: err={understand_err}')

        if understand_obj is None:
            return {
                'status': 'error',
                'type': 'ground-understand',
                'result': '',
                'error': f'Understand failed: {understand_err}',
                'failed_phase': 'understand',
                'bbox_groups': bbox_groups,
            }

        return {
            'status': 'ok',
            'type': 'ground-understand',
            'result': understand_obj.get('result', ''),
            'bbox_groups': bbox_groups,
        }

    # ------------------------------------------------------------------
    # transcribe step (P0-3)
    # ------------------------------------------------------------------

    def _execute_transcribe(self, step, context, image_path, question=''):
        """Execute a transcribe step: full content transcription from image."""
        description = step.get('description', '')
        q_line = f"Original question: {question}\n\n" if question else ""

        messages = [
            {'type': 'text', 'value': (
                f"{q_line}"
                f"Task: {description}\n\n"
                f"{context}\n"
                "Transcribe ALL relevant text content from the image as described "
                "in the task. Output ONLY the JSON object with the result."
            )},
            {'type': 'image', 'value': image_path},
        ]

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = TRANSCRIBE_PROMPT
        try:
            ret, text, _ = self._api_call(
                messages,
                schema=TRANSCRIBE_SCHEMA,
                with_image_processor=True,
                temperature=self.temp_understand,
            )
        finally:
            self.agent.system_prompt = saved_sp

        obj, err = self._parse_json(text, TRANSCRIBE_SCHEMA)
        step_id = step.get('id', 'unknown')
        _log_diag(f'[V5-Transcribe] {step_id}: err={err}')

        if obj is None:
            return {
                'status': 'error',
                'type': 'transcribe',
                'result': '',
                'error': f'Transcribe failed: {err}',
            }

        return {
            'status': 'ok',
            'type': 'transcribe',
            'result': obj.get('result', ''),
        }

    # ------------------------------------------------------------------
    # compute step (text-only)
    # ------------------------------------------------------------------

    def _execute_compute(self, step, context):
        """Execute a compute step (text-only, no image)."""
        description = step.get('description', '')

        user_prompt = (
            f"Task: {description}\n\n"
            f"{context}\n"
            "Perform the calculation described above using the provided data. "
            "Output ONLY the JSON object with the result."
        )

        messages = [{'type': 'text', 'value': user_prompt}]

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = COMPUTE_PROMPT
        try:
            ret, compute_text, _ = self._api_call(
                messages,
                schema=COMPUTE_SCHEMA,
                with_image_processor=False,
                temperature=self.temp_compute,
            )
        finally:
            self.agent.system_prompt = saved_sp

        compute_obj, compute_err = self._parse_json(compute_text, COMPUTE_SCHEMA)
        step_id = step.get('id', 'unknown')
        _log_diag(f'[V5-Compute] {step_id}: err={compute_err}')

        if compute_obj is None:
            return {
                'status': 'error',
                'type': 'compute',
                'result': '',
                'error': f'Compute failed: {compute_err}',
            }

        return {
            'status': 'ok',
            'type': 'compute',
            'result': compute_obj.get('result', ''),
        }

    # ------------------------------------------------------------------
    # logical step (text-only)
    # ------------------------------------------------------------------

    def _execute_logical(self, step, context):
        """Execute a logical step (text-only, no image)."""
        description = step.get('description', '')

        user_prompt = (
            f"Task: {description}\n\n"
            f"{context}\n"
            "Apply the logical condition described above using the provided data. "
            "Output ONLY the JSON object with the result."
        )

        messages = [{'type': 'text', 'value': user_prompt}]

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = LOGICAL_PROMPT
        try:
            ret, logical_text, _ = self._api_call(
                messages,
                schema=LOGICAL_SCHEMA,
                with_image_processor=False,
                temperature=self.temp_logical,
            )
        finally:
            self.agent.system_prompt = saved_sp

        logical_obj, logical_err = self._parse_json(logical_text, LOGICAL_SCHEMA)
        step_id = step.get('id', 'unknown')
        _log_diag(f'[V5-Logical] {step_id}: err={logical_err}')

        if logical_obj is None:
            return {
                'status': 'error',
                'type': 'logical',
                'result': '',
                'error': f'Logical failed: {logical_err}',
            }

        return {
            'status': 'ok',
            'type': 'logical',
            'result': logical_obj.get('result', ''),
        }

    # ------------------------------------------------------------------
    # Stage 3: Reasoning (P0-1 + P2-8)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_reasoning_success(answer_text):
        """Check if reasoning produced a usable answer."""
        if answer_text is None:
            return False
        txt = answer_text.strip()
        return '<a>' in txt and '</a>' in txt

    def _reasoning(self, image_path, question, all_step_results, strategy='',
                   overview=''):
        """Final reasoning stage: TEXT-ONLY synthesis of analysis results.

        P0-1: Separates visual observations (FACT) from derived results (DRAFT).
              Reasoning re-derives compute/logical results from raw observations.
        P2-8: Uses REASONING_SCHEMA for structured JSON output.
        """
        _log_diag('[V5-Stage3] Reasoning started (text-only)')

        # P0-1: Split results by trust level
        obs_lines = []
        derived_lines = []

        if all_step_results:
            for step_id, data in all_step_results.items():
                step_type = data.get('type', 'unknown')
                status = data.get('status', 'unknown')
                desc = data.get('description', '')
                desc_str = f" | task: {desc}" if desc else ""

                if status == 'ok':
                    result = data.get('result', '')
                    if isinstance(result, dict):
                        result = json.dumps(result, ensure_ascii=False)
                    line = f"[{step_id}] ({step_type}{desc_str}): {result}"
                else:
                    err = data.get('error', 'unknown error')
                    line = f"[{step_id}] ({step_type}{desc_str}): FAILED — {err}"

                if step_type in ('ground-understand', 'transcribe'):
                    obs_lines.append(line)
                else:
                    derived_lines.append(line)

        observations_text = "\n".join(obs_lines) if obs_lines else "(No visual observations available.)"
        derived_text = "\n".join(derived_lines) if derived_lines else "(No derived results.)"

        reasoning_prompt = REASONING_PROMPT.format(
            overview=overview or '(No overview available.)',
            strategy=strategy or '(No strategy provided.)',
            observations_text=observations_text,
            derived_text=derived_text,
        )

        user_content = [
            {'type': 'text', 'value': (
                f"Original question: <q>{question}</q>\n\n"
                "Answer based on the information above. "
                "Output ONLY the JSON object with \"reasoning\" and \"answer\" fields."
            )},
        ]

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = reasoning_prompt
        try:
            ret, answer_text, resp = self._api_call(
                user_content,
                schema=REASONING_SCHEMA,
                with_image_processor=False,
                temperature=self.temp_reasoning,
            )
        finally:
            self.agent.system_prompt = saved_sp

        # P2-8: Parse structured output and format with <a> tags
        answer_obj, parse_err = self._parse_json(answer_text, REASONING_SCHEMA)

        if answer_obj and 'answer' in answer_obj:
            reasoning_text = answer_obj.get('reasoning', '')
            answer_val = answer_obj.get('answer', '')
            formatted = f"{reasoning_text}\n\n<a>{answer_val}</a>"
            _log_diag(f'[V5-Stage3] Reasoning OK, answer={answer_val[:100]}')
            return 0, formatted, resp

        _log_diag(f'[V5-Stage3] JSON parse failed ({parse_err}), using raw text')
        return ret, answer_text or '', resp

    # ------------------------------------------------------------------
    # Debug: draw bboxes on image
    # ------------------------------------------------------------------

    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 192, 203),
    ]

    @staticmethod
    def _draw_bboxes_on_image(image_path, bbox_groups, step_id, orig_w, orig_h,
                              output_dir="test_case/v4_test_output"):
        """Draw bounding boxes on the original image and save to disk."""
        os.makedirs(output_dir, exist_ok=True)

        with Image.open(image_path).convert('RGB') as img:
            draw = ImageDraw.Draw(img)
            line_width = max(3, min(orig_w, orig_h) // 400)

            for g_idx, group in enumerate(bbox_groups):
                color = OCRAgentV5.COLORS[g_idx % len(OCRAgentV5.COLORS)]
                group_id = group.get('group_id', 'unknown')

                for b_idx, bbox in enumerate(group.get('bboxes', [])):
                    if len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = bbox
                    px1 = int(x1 / 1000.0 * orig_w)
                    py1 = int(y1 / 1000.0 * orig_h)
                    px2 = int(x2 / 1000.0 * orig_w)
                    py2 = int(y2 / 1000.0 * orig_h)

                    for offset in range(line_width):
                        draw.rectangle(
                            [px1 + offset, py1 + offset, px2 - offset, py2 - offset],
                            outline=color
                        )

                    label = f"{group_id}[{b_idx}]"
                    try:
                        font = ImageFont.truetype("arial.ttf", size=max(10, line_width * 4))
                    except (OSError, IOError):
                        font = ImageFont.load_default()

                    text_h = max(12, line_width * 4)
                    draw.rectangle(
                        [px1, py1 - text_h - 4, px1 + len(label) * 7 + 4, py1],
                        fill=color
                    )
                    draw.text((px1 + 2, py1 - text_h - 2), label, fill=(255, 255, 255), font=font)

            out_path = os.path.join(output_dir, f"locate_{step_id}.jpg")
            img.save(out_path, quality=90)
            _log_diag(f'[V5-BboxDraw] Saved {out_path} ({len(bbox_groups)} groups)')

    # ------------------------------------------------------------------
    # Image cropping utility
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_and_encode(image_path, physical_bbox):
        """Crop a region from the image and return as base64 JPEG."""
        with Image.open(image_path).convert('RGB') as img:
            xmin, ymin, xmax, ymax = physical_bbox

            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(img.width, int(xmax))
            ymax = min(img.height, int(ymax))

            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin

            if xmin == xmax:
                xmax = min(img.width, xmin + 1)
            if ymin == ymax:
                ymax = min(img.height, ymin + 1)

            cropped = img.crop((xmin, ymin, xmax, ymax))
            buffer = io.BytesIO()
            cropped.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_inner(self, inputs, **kwargs):
        """Main inference pipeline: Planning → Executor → Reasoning."""
        import time as _time
        t_start = _time.time()

        image_path = None
        question = ""
        for item in inputs:
            if item['type'] == 'image':
                image_path = item['value']
            elif item['type'] == 'text':
                question += item['value']

        if not image_path:
            return super().generate_inner(inputs, **kwargs)

        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        # ============================================================
        # Stage 1: Planning
        # ============================================================
        t1 = _time.time()
        steps, plan_status, plan_text, strategy = self._planning(image_path, question)
        t1_elapsed = _time.time() - t1

        # ============================================================
        # Stage 2: Execution (normal flow — planning succeeded)
        # ============================================================
        all_step_results = {}
        exec_status = 'success'
        overview = ''

        if plan_status == 'success' and steps:
            overview = self._generate_overview(image_path, question)
            t2 = _time.time()
            all_step_results, exec_status = self._execute_steps(
                steps, image_path, orig_w, orig_h, question, strategy
            )
            t2_elapsed = _time.time() - t2
        else:
            t2_elapsed = 0
            exec_status = plan_status

        # ============================================================
        # Stage 3: Reasoning
        # ============================================================
        t3 = _time.time()

        if plan_status == 'success' and all_step_results:
            ret, answer, resp = self._reasoning(
                image_path, question, all_step_results, strategy, overview=overview
            )
            if not self._is_reasoning_success(answer):
                _log_diag('[V5-Stage3] Reasoning produced no usable answer')
                if exec_status == 'success':
                    exec_status = 'reasoning_failed'
        else:
            ret = -1
            answer = ''
            resp = None

        t3_elapsed = _time.time() - t3
        t_total = _time.time() - t_start

        if plan_status != 'success':
            final_status = plan_status
        elif exec_status != 'success':
            final_status = exec_status
        else:
            final_status = 'success'

        n_steps = len(steps) if steps else 0
        n_ok = sum(1 for r in all_step_results.values() if r.get('status') == 'ok')
        n_err = n_steps - n_ok
        _log_diag(
            f'[V5-Total] status={final_status} | '
            f'steps={n_steps} ok={n_ok} err={n_err} | '
            f'S1={t1_elapsed:.0f}s S2={t2_elapsed:.0f}s S3={t3_elapsed:.0f}s '
            f'total={t_total:.0f}s'
        )

        # ============================================================
        # Build execution trace
        # ============================================================
        trace = self._build_trace(
            image_path=image_path,
            question=question,
            plan_status=plan_status,
            plan_text=plan_text,
            strategy=strategy,
            overview=overview,
            steps=steps,
            t1_elapsed=t1_elapsed,
            all_step_results=all_step_results,
            exec_status=exec_status,
            t2_elapsed=t2_elapsed,
            answer=answer,
            final_status=final_status,
            t3_elapsed=t3_elapsed,
            t_total=t_total,
        )
        self.last_trace = trace

        if self.trace_dir:
            self._write_trace(trace)

        return ret, answer, resp

    # ------------------------------------------------------------------
    # Trace building & persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_step_result(data):
        """Extract a concise summary from a step result dict for the trace."""
        summary = {
            'status': data.get('status', 'unknown'),
            'type': data.get('type', 'unknown'),
            'retry_count': data.get('retry_count', 0),
        }
        if data.get('status') == 'ok':
            result = data.get('result', '')
            if isinstance(result, dict):
                summary['result'] = json.dumps(result, ensure_ascii=False)
            else:
                summary['result'] = str(result)[:500]
        else:
            summary['error'] = data.get('error', '')[:300]

        if data.get('type') == 'ground-understand':
            summary['failed_phase'] = data.get('failed_phase', None)
            bbox_groups = data.get('bbox_groups', None)
            if bbox_groups:
                summary['n_bbox_groups'] = len(bbox_groups)
                summary['n_bboxes'] = sum(
                    len(g.get('bboxes', [])) for g in bbox_groups
                )
            else:
                summary['n_bbox_groups'] = 0
                summary['n_bboxes'] = 0

        return summary

    def _build_trace(self, **kwargs):
        """Build a structured execution trace dict."""
        steps = kwargs['steps']
        all_step_results = kwargs['all_step_results']
        plan_status = kwargs['plan_status']

        planning_trace = {
            'status': plan_status,
            'strategy': kwargs.get('strategy', ''),
            'n_steps': len(steps) if steps else 0,
            'steps': steps,
            'time_s': round(kwargs['t1_elapsed'], 1),
        }

        executor_trace = {
            'overall_status': kwargs['exec_status'] if plan_status == 'success' else plan_status,
            'n_total': len(steps) if steps else 0,
            'n_ok': sum(1 for r in all_step_results.values() if r.get('status') == 'ok'),
            'n_error': sum(1 for r in all_step_results.values() if r.get('status') != 'ok'),
            'time_s': round(kwargs['t2_elapsed'], 1),
            'steps': {},
        }
        for sid in (s['id'] for s in (steps or [])):
            if sid in all_step_results:
                executor_trace['steps'][sid] = self._summarize_step_result(
                    all_step_results[sid]
                )
            else:
                executor_trace['steps'][sid] = {
                    'status': 'not_executed',
                    'type': 'unknown',
                    'retry_count': 0,
                    'error': 'Skipped (dependency or planning failure)',
                }

        answer = kwargs['answer']
        reasoning_trace = {
            'mode': 'text_only',
            'has_answer': '<a>' in (answer or '') and '</a>' in (answer or ''),
            'final_answer': answer[:2000] if answer else None,
            'time_s': round(kwargs['t3_elapsed'], 1),
        }

        return {
            'image_path': kwargs['image_path'],
            'question': kwargs['question'][:500],
            'overall_status': kwargs['final_status'],
            'total_time_s': round(kwargs['t_total'], 1),
            'planning': planning_trace,
            'executor': executor_trace,
            'reasoning': reasoning_trace,
        }

    def _write_trace(self, trace):
        """Append one execution trace as a JSON line to the trace log file."""
        import datetime
        os.makedirs(self.trace_dir, exist_ok=True)
        log_path = os.path.join(self.trace_dir, 'execution_traces.jsonl')
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            **trace,
        }
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
