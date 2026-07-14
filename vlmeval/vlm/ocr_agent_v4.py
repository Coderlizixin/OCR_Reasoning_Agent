"""
OCR-Agent V4: Planning → Executor (DAG-based) → Reasoning Pipeline
====================================================================
All stages use a single Qwen3.5-9B VL model.

Stage 1 - Planning:     image + question → JSON steps with DAG dependencies
Stage 2 - Executor:     execute steps topologically (ground-understand / compute / logical)
Stage 3 - Reasoning:    all step results + original image + question → CoT + final answer

Error handling: no retry; track end status (success | planning_failed | grounding_error | execution_error).
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
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["ground-understand", "compute", "logical"]
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
    "required": ["steps"]
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


# ============================================================================
# System Prompts
# ============================================================================

PLANNING_PROMPT = """You are a visual information collection planner — NOT a problem solver. Your ONLY job is to design a plan for WHAT data to collect from the image and in what order. You will NEVER see the answer or draw conclusions.

## CRITICAL RULES — READ FIRST
1. **DO NOT attempt to answer the question.** You are NOT solving the problem. You are designing a data-collection plan that someone else will execute.
2. **DO NOT form any hypothesis about the answer.** Even if you think you see the answer in the image, ignore it. Your personal observations are irrelevant.
3. **DO NOT include any reasoning, calculation results, or values in your plan.** The step descriptions must state WHAT to look for, never WHAT you think the answer is.
4. **Your output is a "shopping list" of information to extract from the image** — what to read, where to look. Not conclusions.
5. **Do NOT output phrases like 'let me check again', 'wait', 'perhaps', or 'it seems'. Output your calculations with absolute certainty.**

## Step Types
- **ground-understand**: Extract a specific piece of information from the image (e.g., read a labeled value, identify what a legend says, find what number a bar corresponds to). The executor will locate the region and read it.
- **compute**: Perform a calculation using results from previous steps. No image access needed.
- **logical**: Make a judgment, comparison, or filter using results from previous steps. No image access needed.

## Rules for building the plan
1. Each step must accomplish exactly ONE atomic information-gathering task.
2. Order steps so raw data is collected first (ground-understand), then processed (compute/logical).
3. `depends_on` must ONLY reference step IDs that appear earlier.
4. `depends_on` should be [] for steps that don't need prior results (typically initial ground-understand steps).
5. Be specific in descriptions — name exact labels, positions, or visual features to locate. Say "Find the number next to the label 'Q1' in the chart" not "Read Q1 value".

## Output Format
You MUST output a single JSON object. NO other text before or after.

{"steps": [{"id": "...", "type": "ground-understand|compute|logical", "description": "...", "depends_on": ["..."]}]}

## Examples of GOOD vs BAD step descriptions

BAD: "Q1 sales is 42.5, Q2 sales is 38.2, sum them"          ← this is answering, not planning!
BAD: "Calculate the total since it will exceed 1 million"      ← this assumes an answer!
GOOD: "Find the sales value for Q1 shown in the chart"          ← pure information-gathering
GOOD: "Calculate the sum of Q1 and Q2 sales values"             ← describes computation, not the result

## Example Plan (correct style)
{
  "steps": [
    {"id": "step_1", "type": "ground-understand", "description": "Find the value labeled or corresponding to Q1 in the chart", "depends_on": []},
    {"id": "step_2", "type": "ground-understand", "description": "Find the value labeled or corresponding to Q2 in the chart", "depends_on": []},
    {"id": "step_3", "type": "compute", "description": "Calculate the sum of Q1's value and Q2's value", "depends_on": ["step_1", "step_2"]},
    {"id": "step_4", "type": "logical", "description": "Compare the sum against 1 million to determine if it exceeds this threshold", "depends_on": ["step_3"]}
  ]
}

REMEMBER: You are a planner, not a solver. If you catch yourself thinking about what the answer might be, STOP. Just list what data to collect."""

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
3. Return the judgment or filtered results as a self-contained descriptive sentence.

## Output Format
You MUST output a single JSON object: {"result": "your result here"}

## CRITICAL — Complete Sentences Required
Your "result" MUST be a **self-contained descriptive sentence**. Never output a bare boolean or bare list.

GOOD examples:
- "The total of 80.7 does not exceed 100, so the condition is NOT met"
- "The products with sales exceeding 50 are: Product A (65), Product C (72)"
- "Q1 (42.5) is higher than Q2 (38.2), so Q1 is the larger quarter"

BAD examples (FORBIDDEN):
- "true"           ← bare boolean
- "False"          ← bare boolean
- "['Product A', 'Product C']"   ← bare list without context"""

REASONING_PROMPT = """You are a final-answer synthesizer. You are given VERIFIED analysis results from an automated extraction pipeline that already read the image. You do NOT have access to the image — you ONLY have the text results below.

## VERIFIED Analysis Results (GROUND TRUTH — accept as FACT)
{analysis_text}

## ABSOLUTE RULES — VIOLATION WILL CAUSE WRONG ANSWERS
1. **The analysis results are GROUND TRUTH.** They were extracted by a separate system that already looked at the image. You CANNOT verify or double-check them. Accept them as FACT without question.
2. **You have NO access to any image.** There is no image for you to look at. You MUST work exclusively from the text results above.
3. **If a value is stated, USE IT directly.** Do not second-guess, recalculate from scratch, or wonder if it might be wrong.
4. **If some steps FAILED, note which data is missing.** State clearly "step X failed to extract [value], so [what is missing]". Do NOT attempt to fill gaps by guessing.
5. **Under 200 words total.**
6. **FORBIDDEN phrases:** "let me check", "wait", "looking at the image", "I can see", "the image shows", "let me verify", "let me re-examine", "perhaps", "maybe". Using any of these means you are disobeying Rule 2.

## OUTPUT
1. Identify which analysis result(s) answer the question.
2. If the answer is directly in the results, output it.
3. If data is missing due to a failed step, state what is missing and give the best possible answer from available data.
4. Enclose your FINAL answer in <a> and </a> tags. Example: <a>24</a> or <a>21 percent</a>"""


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

class OCRAgentV4(OpenAIWrapper):
    """V4 OCR Agent: Planning → Executor (DAG) → Reasoning.

    All three stages use the same 9B VL model. The executor runs steps
    in topological order, each as an independent API call with its own
    system prompt and (optionally) constrained JSON decoding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Internal agent for all API calls — shares model, api_base, key, etc.
        # Uses _Base64Agent so we can pass cropped images as base64 in-memory.
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
            'retry': 3,     # fewer retries for internal calls
            'wait': 3,
        }
        self.agent = _Base64Agent(**agent_kwargs)

    # ------------------------------------------------------------------
    # Core API helper
    # ------------------------------------------------------------------

    def _api_call(self, messages, schema=None, with_image_processor=False):
        """Make a single API call with optional guided JSON decoding.

        Args:
            messages: Flat list of {'type','value'} dicts OR list of
                      {'role','content'} dicts for multi-turn.
            schema: Optional JSON Schema for vLLM guided_json.
            with_image_processor: If True, include mm_processor_kwargs
                                  from self.extra_body for high-res images.

        Returns:
            (ret_code: int, text: str, response_obj)
        """
        saved_extra = self.agent.extra_body

        # Build extra_body for this specific call
        new_extra = {}
        if with_image_processor and self.extra_body:
            new_extra.update(self.extra_body)
        if schema is not None:
            new_extra['guided_json'] = schema

        self.agent.extra_body = new_extra if new_extra else None
        try:
            return self.agent.generate_inner(messages)
        finally:
            self.agent.extra_body = saved_extra

    # ------------------------------------------------------------------
    # JSON parsing with fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text, schema=None):
        """Try to parse JSON from model output. Returns (obj, error_msg)."""
        if text is None:
            return None, "text is None"

        text = text.strip()

        # Direct parse
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            pass

        # Strip thinking traces — the model may output <｜end▁of▁thinking｜>...<｜end▁of▁thinking｜> before JSON
        # Extract content after the LAST  response or  response tag
        for end_tag in ['</think>', '<|end_of_thinking|>']:
            if end_tag in text:
                after = text.rsplit(end_tag, 1)[-1].strip()
                try:
                    return json.loads(after), None
                except json.JSONDecodeError:
                    pass
                # If direct parse fails, replace the text with the extracted portion
                # and continue to fence/brace extraction below
                text = after
                break

        # Try to extract JSON block from markdown fences
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip()), None
            except json.JSONDecodeError:
                pass

        # Try to find the outermost { } or [ ]
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
    # Context assembly for step execution
    # ------------------------------------------------------------------

    @staticmethod
    def _build_step_context(step, all_step_results):
        """Build a context string from the dependent steps' results.

        Args:
            step: The step dict with 'depends_on' list of step IDs.
            all_step_results: Dict mapping step_id -> result_data dict.
                Each result_data has: {'status': 'ok'|'error', 'result': str|dict}

        Returns:
            A formatted string of dependent step results, or empty string.
        """
        dep_ids = step.get('depends_on', [])
        if not dep_ids:
            return ""

        lines = ["Here are the results of previous steps that this step depends on:\n"]
        for dep_id in dep_ids:
            dep_result = all_step_results.get(dep_id, {})
            if dep_result.get('status') == 'ok':
                result_val = dep_result.get('result', '')
                if isinstance(result_val, dict):
                    result_val = json.dumps(result_val, ensure_ascii=False)
                lines.append(
                    f"[Step {dep_id} ({dep_result.get('type', 'unknown')})]:\n"
                    f"{result_val}\n"
                )
            else:
                lines.append(
                    f"[Step {dep_id}]: FAILED — {dep_result.get('error', 'unknown error')}\n"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 1: Planning
    # ------------------------------------------------------------------

    def _planning(self, image_path, question):
        """Generate the execution plan (DAG of steps).

        Returns:
            (steps: list | None, status: str, plan_text: str)
        """
        _log_diag('[V4-Stage1] Planning started')

        user_prompt = (
            f"Question that will eventually need to be answered: <q>{question}</q>\n\n"
            "Your task is NOT to answer this question. Your task is to design a "
            "data-collection plan: what information needs to be extracted from the "
            "image, in what order, so that someone else can later answer the question.\n\n"
            "Look at the image ONLY to understand what visual elements exist — NOT to "
            "compute any answer. Then output the JSON plan of ground-understand / compute "
            "/ logical steps. NO extra text."
        )

        messages = [
            {'type': 'text', 'value': user_prompt},
            {'type': 'image', 'value': image_path},
        ]

        # Use system_prompt on agent temporarily for the system message
        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = PLANNING_PROMPT
        try:
            ret, plan_text, resp = self._api_call(
                messages,
                schema=PLANNING_SCHEMA,
                with_image_processor=True,
            )
        finally:
            self.agent.system_prompt = saved_sp

        _log_diag(f'[V4-Stage1] Planning ret={ret}, text_len={len(plan_text) if plan_text else 0}')

        if ret != 0 or not plan_text or self.fail_msg in plan_text:
            return None, 'planning_failed', plan_text or ''

        plan_obj, err = self._parse_json(plan_text, PLANNING_SCHEMA)
        if plan_obj is None or 'steps' not in plan_obj:
            _log_diag(f'[V4-Stage1] JSON parse error: {err}')
            return None, 'planning_failed', plan_text

        steps = plan_obj['steps']
        # Validate steps
        seen_ids = set()
        for s in steps:
            sid = s.get('id', '')
            if not sid or sid in seen_ids:
                _log_diag(f'[V4-Stage1] Invalid step id: {sid}')
                return None, 'planning_failed', plan_text
            seen_ids.add(sid)
            # Check depends_on only references seen IDs
            for dep in s.get('depends_on', []):
                if dep not in seen_ids:
                    _log_diag(f'[V4-Stage1] Step {sid} depends on unseen step {dep}')
                    return None, 'planning_failed', plan_text

        _log_diag(f'[V4-Stage1] Planning OK — {len(steps)} steps')
        return steps, 'success', plan_text

    # ------------------------------------------------------------------
    # Stage 2: Step Execution
    # ------------------------------------------------------------------

    def _execute_steps(self, steps, image_path, orig_w, orig_h):
        """Execute steps in order (already topologically sorted).

        Args:
            steps: List of step dicts.
            image_path: Path to the original image.
            orig_w, orig_h: Original image dimensions.

        Returns:
            (all_step_results: dict, overall_status: str)
        """
        all_results = {}
        overall_status = 'success'

        for step in steps:
            step_id = step.get('id', 'unknown')
            step_type = step.get('type', 'unknown')
            _log_diag(f'[V4-Stage2] Executing {step_id} ({step_type})')

            # Build context from dependencies
            context = self._build_step_context(step, all_results)

            try:
                if step_type == 'ground-understand':
                    result_data = self._execute_ground_understand(
                        step, context, image_path, orig_w, orig_h
                    )
                elif step_type == 'compute':
                    result_data = self._execute_compute(step, context)
                elif step_type == 'logical':
                    result_data = self._execute_logical(step, context)
                else:
                    result_data = {
                        'status': 'error',
                        'type': step_type,
                        'error': f'Unknown step type: {step_type}',
                        'result': '',
                    }
            except Exception as e:
                result_data = {
                    'status': 'error',
                    'type': step_type,
                    'error': str(e),
                    'result': '',
                }

            all_results[step_id] = result_data

            if result_data.get('status') == 'error':
                if step_type == 'ground-understand':
                    if overall_status == 'success':
                        overall_status = 'grounding_error'
                else:
                    if overall_status == 'success':
                        overall_status = 'execution_error'

        return all_results, overall_status

    # ------------------------------------------------------------------
    # ground-understand: Phase 1 (Locate) + Phase 2 (Understand)
    # ------------------------------------------------------------------

    def _execute_ground_understand(self, step, context, image_path, orig_w, orig_h):
        """Execute a ground-understand step: locate → crop → understand.

        Returns:
            {'status': 'ok'|'error', 'type': 'ground-understand',
             'result': str, 'bbox_groups': [...], 'error': str}
        """
        description = step.get('description', '')

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
            )
        finally:
            self.agent.system_prompt = saved_sp

        locate_obj, locate_err = self._parse_json(locate_text, LOCATE_SCHEMA)
        step_id = step.get('id', 'unknown')
        _log_diag(f'[V4-Locate] {step_id}: err={locate_err}, obj_keys={list(locate_obj.keys()) if locate_obj else None}')

        if locate_obj is None or not locate_obj.get('bbox_groups'):
            return {
                'status': 'error',
                'type': 'ground-understand',
                'result': '',
                'error': f'Locate failed: {locate_err or "empty bbox_groups"}',
            }

        bbox_groups = locate_obj['bbox_groups']

        # Debug: draw bboxes on the original image
        self._draw_bboxes_on_image(
            image_path, bbox_groups, step_id, orig_w, orig_h
        )

        # Collect all valid bboxes first; skip if none are usable
        all_crops = []  # (group_id, purpose, bbox_idx, physical_bbox)
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

        # Group crops by group_id for labeling
        current_group = None
        for group_id, purpose, bbox_idx, phys_bbox in all_crops:
            # Emit group header when switching groups
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
                with_image_processor=False,  # crops are small, no need for high-res
            )
        finally:
            self.agent.system_prompt = saved_sp

        understand_obj, understand_err = self._parse_json(understand_text, UNDERSTAND_SCHEMA)
        _log_diag(f'[V4-Understand] {step.get("id")}: err={understand_err}')

        if understand_obj is None:
            return {
                'status': 'error',
                'type': 'ground-understand',
                'result': '',
                'error': f'Understand failed: {understand_err}',
                'bbox_groups': bbox_groups,
            }

        return {
            'status': 'ok',
            'type': 'ground-understand',
            'result': understand_obj.get('result', ''),
            'bbox_groups': bbox_groups,
        }

    # ------------------------------------------------------------------
    # compute step (text-only)
    # ------------------------------------------------------------------

    def _execute_compute(self, step, context):
        """Execute a compute step (text-only, no image).

        Returns:
            {'status': 'ok'|'error', 'type': 'compute', 'result': str, 'error': str}
        """
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
            )
        finally:
            self.agent.system_prompt = saved_sp

        compute_obj, compute_err = self._parse_json(compute_text, COMPUTE_SCHEMA)
        _log_diag(f'[V4-Compute] {step.get("id")}: err={compute_err}')

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
        """Execute a logical step (text-only, no image).

        Returns:
            {'status': 'ok'|'error', 'type': 'logical', 'result': ..., 'error': str}
        """
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
            )
        finally:
            self.agent.system_prompt = saved_sp

        logical_obj, logical_err = self._parse_json(logical_text, LOGICAL_SCHEMA)
        _log_diag(f'[V4-Logical] {step.get("id")}: err={logical_err}')

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
    # Stage 3: Reasoning
    # ------------------------------------------------------------------

    def _reasoning(self, image_path, question, all_step_results):
        """Final reasoning stage: TEXT-ONLY synthesis of analysis results.

        The original image is deliberately NOT shown to the model here.
        This forces the model to trust the executor results instead of
        re-reading the image. If ground-understand steps failed, the
        caller should route to _fallback_reasoning instead.

        Returns:
            (ret_code, answer_text, response_obj)
        """
        _log_diag('[V4-Stage3] Reasoning started (text-only, no image)')

        # Build analysis summary
        if all_step_results:
            lines = []
            for step_id, data in all_step_results.items():
                step_type = data.get('type', 'unknown')
                status = data.get('status', 'unknown')
                if status == 'ok':
                    result = data.get('result', '')
                    if isinstance(result, dict):
                        result = json.dumps(result, ensure_ascii=False)
                    lines.append(f"[{step_id}] ({step_type}): {result}")
                else:
                    err = data.get('error', 'unknown error')
                    lines.append(f"[{step_id}] ({step_type}): FAILED — {err}")
            analysis_text = "\n".join(lines)
        else:
            analysis_text = "(No analysis results available.)"

        reasoning_prompt = REASONING_PROMPT.format(analysis_text=analysis_text)

        # Text-only — no image. The model can only use the analysis results.
        user_content = [
            {'type': 'text', 'value': (
                f"Original question: <q>{question}</q>\n\n"
                "You have NO image. Use ONLY the analysis results above. "
                "If data is missing due to FAILED steps, say so. "
                "Please provide the final answer FIRST, enclosed strictly within the tags  and  like this: your <a> answer </a>. "
                "ONLY AFTER providing the answer, you may briefly explain your reasoning. "
                "Enclose your final answer in <a> and </a> tags."
            )},
        ]

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = reasoning_prompt
        try:
            ret, answer, resp = self._api_call(
                user_content,
                schema=None,  # no guided decoding — free-form reasoning
                with_image_processor=False,
            )
        finally:
            self.agent.system_prompt = saved_sp

        _log_diag(f'[V4-Stage3] Reasoning ret={ret}, answer_len={len(answer) if answer else 0}')
        return ret, answer, resp

    # ------------------------------------------------------------------
    # Fallback: direct reasoning (when planning fails)
    # ------------------------------------------------------------------

    def _fallback_reasoning(self, image_path, question):
        """Direct reasoning when planning fails — clean slate, image only."""
        _log_diag('[V4-Fallback] Planning failed, falling back to direct image reasoning')

        prompt = (
            f"Question: <q>{question}</q>\n\n"
            "Look at the image carefully and answer the question step by step. "
            "Enclose your final answer in <a> and </a> tags."
        )

        messages = [
            {'type': 'text', 'value': prompt},
            {'type': 'image', 'value': image_path},
        ]

        ret, answer, resp = self._api_call(
            messages,
            schema=None,
            with_image_processor=True,  # high-res for reading small text/numbers
        )
        return ret, answer, resp

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
        """Draw bounding boxes on the original image and save to disk.

        Args:
            image_path: Path to the original image.
            bbox_groups: List of bbox group dicts from Locate phase.
            step_id: Step ID for the output filename.
            orig_w, orig_h: Original image dimensions.
            output_dir: Directory to save the annotated image.
        """
        os.makedirs(output_dir, exist_ok=True)

        with Image.open(image_path).convert('RGB') as img:
            draw = ImageDraw.Draw(img)
            line_width = max(3, min(orig_w, orig_h) // 400)

            for g_idx, group in enumerate(bbox_groups):
                color = OCRAgentV4.COLORS[g_idx % len(OCRAgentV4.COLORS)]
                group_id = group.get('group_id', 'unknown')
                purpose = group.get('purpose', '')[:80]

                for b_idx, bbox in enumerate(group.get('bboxes', [])):
                    if len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = bbox
                    px1 = int(x1 / 1000.0 * orig_w)
                    py1 = int(y1 / 1000.0 * orig_h)
                    px2 = int(x2 / 1000.0 * orig_w)
                    py2 = int(y2 / 1000.0 * orig_h)

                    # Draw rectangle
                    for offset in range(line_width):
                        draw.rectangle(
                            [px1 + offset, py1 + offset, px2 - offset, py2 - offset],
                            outline=color
                        )

                    # Draw label: "group_id (idx): purpose"
                    label = f"{group_id}[{b_idx}]"
                    bbox_text = f"[{x1},{y1},{x2},{y2}]"
                    try:
                        font = ImageFont.truetype("arial.ttf", size=max(10, line_width * 4))
                    except (OSError, IOError):
                        font = ImageFont.load_default()

                    # Background for label text
                    text_h = max(12, line_width * 4)
                    draw.rectangle(
                        [px1, py1 - text_h - 4, px1 + len(label) * 7 + 4, py1],
                        fill=color
                    )
                    draw.text((px1 + 2, py1 - text_h - 2), label, fill=(255, 255, 255), font=font)

            out_path = os.path.join(output_dir, f"locate_{step_id}.jpg")
            img.save(out_path, quality=90)
            _log_diag(f'[V4-BboxDraw] Saved {out_path} ({len(bbox_groups)} groups)')

    # ------------------------------------------------------------------
    # Image cropping utility
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_and_encode(image_path, physical_bbox):
        """Crop a region from the image and return as base64 JPEG.

        Args:
            image_path: Path to the image file.
            physical_bbox: (xmin, ymin, xmax, ymax) in pixel coordinates.

        Returns:
            Base64-encoded JPEG string.
        """
        with Image.open(image_path).convert('RGB') as img:
            xmin, ymin, xmax, ymax = physical_bbox

            # Clamp to image bounds
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(img.width, int(xmax))
            ymax = min(img.height, int(ymax))

            # Swap if inverted
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin

            # Prevent zero-size
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
        """Main inference pipeline.

        inputs: list of {'type': 'text'|'image', 'value': ...}
        """
        import time as _time
        t_start = _time.time()

        # Extract image and question
        image_path = None
        question = ""
        for item in inputs:
            if item['type'] == 'image':
                image_path = item['value']
            elif item['type'] == 'text':
                question += item['value']

        # No image → fall back to standard API call
        if not image_path:
            return super().generate_inner(inputs, **kwargs)

        # Get original image dimensions
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        # ============================================================
        # Stage 1: Planning
        # ============================================================
        t1 = _time.time()
        steps, plan_status, plan_text = self._planning(image_path, question)
        t1_elapsed = _time.time() - t1

        # ============================================================
        # Stage 2: Execution (if planning succeeded)
        # ============================================================
        all_step_results = {}
        exec_status = 'success'

        if plan_status == 'success' and steps:
            t2 = _time.time()
            all_step_results, exec_status = self._execute_steps(
                steps, image_path, orig_w, orig_h
            )
            t2_elapsed = _time.time() - t2
        else:
            t2_elapsed = 0
            exec_status = plan_status  # 'planning_failed'

        # ============================================================
        # Stage 3: Reasoning
        # ============================================================
        t3 = _time.time()

        if plan_status == 'success' and all_step_results:
            # Text-only synthesis — model has NO access to the image.
            # It must trust the step results (even partial ones) and
            # cannot re-verify, so partial success is still useful.
            ret, answer, resp = self._reasoning(
                image_path, question, all_step_results
            )
        else:
            # Planning failed — fall back to direct image reasoning
            ret, answer, resp = self._fallback_reasoning(image_path, question)

        t3_elapsed = _time.time() - t3
        t_total = _time.time() - t_start

        # Determine final status
        if plan_status == 'planning_failed':
            final_status = 'planning_failed'
        elif exec_status != 'success':
            final_status = exec_status  # grounding_error or execution_error
        else:
            final_status = 'success'

        # Log timing and status
        n_steps = len(steps) if steps else 0
        n_ok = sum(1 for r in all_step_results.values() if r.get('status') == 'ok')
        n_err = n_steps - n_ok
        _log_diag(
            f'[V4-Total] status={final_status} | '
            f'steps={n_steps} ok={n_ok} err={n_err} | '
            f'S1={t1_elapsed:.0f}s S2={t2_elapsed:.0f}s S3={t3_elapsed:.0f}s '
            f'total={t_total:.0f}s'
        )

        return ret, answer, resp
