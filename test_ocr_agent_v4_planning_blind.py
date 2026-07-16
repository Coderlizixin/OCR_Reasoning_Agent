"""
独立测试：验证 Planning 是否是"看图反推"还是"真正基于数据规划"。

核心问题: V4 的 Planning 阶段同时接收 image + question，模型可能先"看出"了答案，
然后反推出一个看似合理的执行计划。这个测试将 Planning 与图像隔离:

  Stage 0 - Visual Extraction:  image → JSON (纯文字提取，不许推理)
  Stage 1 - Planning (blind):   JSON → DAG steps (无图像，只能基于提取的文字做规划)
  Stage 2 - Executor:           按拓扑序执行 step
  Stage 3 - Reasoning:          text-only 最终推理

对比:
  - 原始 V4:  Planning(image+question) → Executor → Reasoning
  - Blind V4:  Extraction(image) → Planning(json) → Executor → Reasoning

如果 Blind Planning 能给出同样合理的 DAG，说明 Planning 确实是数据驱动的；
如果 Blind Planning 给出的计划不合理，说明原版 Planning 依赖了图像中的"直觉"。

用法:
    python test_ocr_agent_v4_planning_blind.py
"""
import os
import sys
import io
import re
import json
import base64
import copy
import time as _time
from PIL import Image, ImageDraw, ImageFont

from vlmeval.api.gpt import OpenAIWrapper

# ============================================================================
# 配置
# ============================================================================
LOCAL_IMAGE_PATH = "test_case/test_images/image4.jpg"
QUESTION = "How many are somewhat confident that Donald Trump can manage the executive branch effectively?"
API_BASE = "http://10.3.112.34:22630/v1/chat/completions"
API_KEY = "sk-EMPTY"
MODEL_NAME = "Qwen3.5-9B"
MAX_TOKENS = 6144
OUTPUT_DIR = "test_case/v4_test_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}")


# ============================================================================
# 内部 Agent: 支持 base64 图片
# ============================================================================
class _Base64Agent(OpenAIWrapper):
    """支持 base64 编码图片的 OpenAIWrapper 子类"""

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
# JSON Schemas
# ============================================================================
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "image_description": {"type": "string"},
        "visual_elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                    "location_hint": {"type": "string"}
                },
                "required": ["label", "value"]
            }
        },
        "text_content": {"type": "string"},
        "numerical_data": {"type": "string"},
        "structural_notes": {"type": "string"}
    },
    "required": ["image_description", "visual_elements", "text_content", "numerical_data"]
}

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
    "properties": {"result": {"type": "string"}},
    "required": ["result"]
}

COMPUTE_SCHEMA = {
    "type": "object",
    "properties": {"result": {"type": "string"}},
    "required": ["result"]
}

LOGICAL_SCHEMA = {
    "type": "object",
    "properties": {"result": {}},
    "required": ["result"]
}


# ============================================================================
# System Prompts
# ============================================================================

EXTRACTION_PROMPT = """You are a precise visual transcriber. Your ONLY job is to read and record ALL text, numbers, labels, and structural information visible in the image — as a machine-readable JSON document.

## ABSOLUTE RULES
1. **TRANSCRIBE ONLY.** Do NOT analyze, interpret, compare, calculate, or draw conclusions.
2. **Do NOT answer any question.** You are not given a question. You are only given an image.
3. **Record EVERYTHING visible** — every label, number, legend entry, axis tick, title, annotation. Assume anything you omit is lost forever.
4. **Be faithful to what you see.** If a number says "42.5", record "42.5". Do not round, estimate, or "correct" it.
5. **Location hints**: for each visual element, briefly note where it is (e.g. "top-left legend", "x-axis label", "blue bar", "rightmost bar label") so a planner can later use this to decide what to extract.
6. **Do NOT use thinking mode heavily.** Think in 1-2 sentences, then output the JSON immediately.

## Output Format
You MUST output a single JSON object. NO other text before or after.

{
  "image_description": "Brief 1-sentence description of what the image shows (e.g. 'A grouped bar chart comparing confidence levels across multiple categories')",
  "visual_elements": [
    {"label": "...", "value": "...", "location_hint": "..."}
  ],
  "text_content": "All text visible in the image, transcribed verbatim, preserving hierarchy and position. Include titles, subtitles, axis labels, legend text, annotations, footnotes — EVERYTHING.",
  "numerical_data": "All numerical values found, with their labels/context. Format: one per line like 'Q1 sales (blue bar): 42.5'",
  "structural_notes": "Describe the chart/graphic structure: type of visualization, axes, legends, color coding, groupings. This helps a planner understand how data is organized."
}

## Example
For a bar chart titled "2023 Sales by Quarter" with Q1=42.5, Q2=38.2, Q3=51.0, Q4=45.8:
{
  "image_description": "A vertical bar chart showing 2023 quarterly sales figures with 4 bars in different colors",
  "visual_elements": [
    {"label": "Title", "value": "2023 Sales by Quarter", "location_hint": "top center of chart"},
    {"label": "Q1", "value": "42.5", "location_hint": "first blue bar, x-axis label Q1"},
    {"label": "Q2", "value": "38.2", "location_hint": "second green bar, x-axis label Q2"},
    {"label": "Q3", "value": "51.0", "location_hint": "third red bar, x-axis label Q3"},
    {"label": "Q4", "value": "45.8", "location_hint": "fourth orange bar, x-axis label Q4"},
    {"label": "Y-axis", "value": "0 to 60, step 10", "location_hint": "left side vertical axis"},
    {"label": "Units", "value": "10,000 RMB", "location_hint": "below y-axis or in subtitle"}
  ],
  "text_content": "Title: 2023 Sales by Quarter\\nY-axis: 0, 10, 20, 30, 40, 50, 60\\nX-axis: Q1, Q2, Q3, Q4\\nSubtitle: Units: 10,000 RMB",
  "numerical_data": "Q1 (blue bar): 42.5\\nQ2 (green bar): 38.2\\nQ3 (red bar): 51.0\\nQ4 (orange bar): 45.8\\nY-axis range: 0-60",
  "structural_notes": "Vertical bar chart. 4 bars in different colors. Y-axis on left (0-60, step 10). X-axis labels: Q1-Q4. Each bar has a data label on top. Legend may or may not be present."
}"""

BLIND_PLANNING_PROMPT = """You are an information-collection planner. You are given a JSON document that describes ALL the content of an image — its text, numbers, structure, and visual elements. You do NOT have access to the image itself. You MUST plan exclusively from the provided JSON data.

## CRITICAL RULES
1. **You CANNOT see the image.** Your only source of information is the JSON document below.
2. **DO NOT guess or hallucinate** data not present in the JSON. If the JSON doesn't mention something, assume it doesn't exist.
3. **DO NOT attempt to answer the question yourself.** You are a planner, not a solver. You design a data-collection plan for an executor.
4. **DO NOT include reasoning results, calculation results, or answer values in your plan.** Step descriptions must state WHAT to extract or compute, never WHAT the result is.
5. **Use `location_hint` from the JSON** to write specific step descriptions (e.g. "Find the number next to the 'Q1' label on the first blue bar" rather than "Read Q1 value").

## Step Types
- **ground-understand**: Extract a specific piece of information from the image. The executor will locate and read it.
- **compute**: Perform a calculation using results from previous steps. No image access.
- **logical**: Make a judgment, comparison, or filter using results from previous steps.

## Rules for building the plan
1. Each step must accomplish exactly ONE atomic information-gathering task.
2. Order steps so raw data is collected first (ground-understand), then processed (compute/logical).
3. `depends_on` must ONLY reference step IDs that appear earlier.
4. Be specific in descriptions — reference exact labels and locations from the JSON.

## Output Format
You MUST output a single JSON object. NO other text.

{"steps": [{"id": "...", "type": "ground-understand|compute|logical", "description": "...", "depends_on": ["..."]}]}

## GOOD vs BAD
BAD:  "Q1 is 42.5, Q2 is 38.2"  ← this is answering, not planning!
BAD:  "Calculate the total since it exceeds 1M"  ← assumes an answer
GOOD: "Extract the numerical value for Q1 from the blue bar in the chart"
GOOD: "Sum the Q1 and Q2 values obtained earlier"

REMEMBER: You are a planner working from a text document. You cannot see any image. Trust ONLY the JSON data provided."""

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
# Pipeline 类
# ============================================================================

class BlindPlanningPipeline:
    """四阶段 Pipeline: Extraction → Planning(blind) → Executor → Reasoning"""

    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 192, 203),
    ]

    def __init__(self):
        common = dict(
            model=MODEL_NAME, api_base=API_BASE, key=API_KEY,
            temperature=0.0, max_tokens=MAX_TOKENS, img_size=-1,
            img_detail='high', verbose=True, retry=3, wait=3,
            extra_body={
                "mm_processor_kwargs": {
                    "images_kwargs": {
                        "size": {
                            "longest_edge": 8192 * 32 * 32,
                            "shortest_edge": 4096 * 32 * 32,
                        }
                    }
                }
            },
        )

        # 原始 V4 agent (用于对比)
        from vlmeval.vlm.ocr_agent_v4 import OCRAgentV4
        self.v4_agent_ctor = OCRAgentV4
        self.common_kwargs = common

        # 内部 agent
        self.agent = _Base64Agent(**common)

    # ------------------------------------------------------------------
    # JSON 解析
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(text, schema=None):
        if text is None:
            return None, "text is None"
        text = text.strip()
        try:
            return json.loads(text), None
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
    # API 调用
    # ------------------------------------------------------------------
    def _api_call(self, messages, schema=None, with_image_processor=False):
        saved_extra = self.agent.extra_body
        new_extra = {}
        if with_image_processor and self.agent.extra_body:
            new_extra.update(self.agent.extra_body)
        if schema is not None:
            new_extra['guided_json'] = schema
        self.agent.extra_body = new_extra if new_extra else None
        try:
            return self.agent.generate_inner(messages)
        finally:
            self.agent.extra_body = saved_extra

    # ------------------------------------------------------------------
    # 图像裁剪 & bbox工具
    # ------------------------------------------------------------------
    @staticmethod
    def _crop_and_encode(image_path, physical_bbox):
        with Image.open(image_path).convert('RGB') as img:
            xmin, ymin, xmax, ymax = physical_bbox
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(img.width, int(xmax))
            ymax = min(img.height, int(ymax))
            if xmin > xmax: xmin, xmax = xmax, xmin
            if ymin > ymax: ymin, ymax = ymax, ymin
            if xmin == xmax: xmax = min(img.width, xmin + 1)
            if ymin == ymax: ymax = min(img.height, ymin + 1)
            cropped = img.crop((xmin, ymin, xmax, ymax))
            buffer = io.BytesIO()
            cropped.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _draw_bboxes(self, image_path, bbox_groups, step_id, orig_w, orig_h):
        """在图上画 bbox 并保存"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with Image.open(image_path).convert('RGB') as img:
            draw = ImageDraw.Draw(img)
            line_width = max(3, min(orig_w, orig_h) // 400)
            for g_idx, group in enumerate(bbox_groups):
                color = self.COLORS[g_idx % len(self.COLORS)]
                group_id = group.get('group_id', 'unknown')
                for b_idx, bbox in enumerate(group.get('bboxes', [])):
                    if len(bbox) != 4: continue
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
            out_path = os.path.join(OUTPUT_DIR, f"locate_{step_id}.jpg")
            img.save(out_path, quality=90)
            log(f"  [Bbox] → {out_path}")

    # ------------------------------------------------------------------
    # 上下文组装
    # ------------------------------------------------------------------
    @staticmethod
    def _build_step_context(step, all_step_results):
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
                    f"[Step {dep_id} ({dep_result.get('type', 'unknown')})]:\n{result_val}\n"
                )
            else:
                lines.append(
                    f"[Step {dep_id}]: FAILED — {dep_result.get('error', 'unknown error')}\n"
                )
        return "\n".join(lines)

    # ==================================================================
    # Stage 0: Visual Extraction (image → JSON)
    # ==================================================================
    def stage0_extraction(self, image_path):
        """纯文字提取：看图输出结构化JSON，不做任何推理"""
        log("=" * 60)
        log("[Stage 0] Visual Extraction — pure transcription, no reasoning")
        log("=" * 60)

        user_prompt = (
            "Transcribe ALL visible content from this image into the JSON format specified. "
            "Record every label, number, text, legend, axis tick, and structural detail. "
            "Do NOT analyze, compute, or draw conclusions. Be a faithful transcriber. "
            "Do NOT answer any question — there is no question. Just transcribe."
        )

        messages = [
            {'type': 'text', 'value': user_prompt},
            {'type': 'image', 'value': image_path},
        ]

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = EXTRACTION_PROMPT
        try:
            ret, text, resp = self._api_call(
                messages, schema=EXTRACTION_SCHEMA, with_image_processor=True
            )
        finally:
            self.agent.system_prompt = saved_sp

        extraction_obj, err = self._parse_json(text)
        if extraction_obj is None:
            log(f"[Stage 0] FAILED: {err}")
            log(f"[Stage 0] Raw output (first 500 chars):\n{text[:500] if text else 'None'}")
            return None, text or ''

        log(f"[Stage 0] OK — {len(extraction_obj.get('visual_elements', []))} elements extracted")
        return extraction_obj, text

    # ==================================================================
    # Stage 1: Blind Planning (JSON → DAG, 无图像)
    # ==================================================================
    def stage1_blind_planning(self, extraction_json, question):
        """仅基于提取的JSON文字来规划，看不到图像"""
        log("=" * 60)
        log("[Stage 1] Blind Planning — JSON only, NO image access")
        log("=" * 60)

        extraction_text = json.dumps(extraction_json, ensure_ascii=False, indent=2)

        user_prompt = (
            f"## Extracted Image Content (YOUR ONLY DATA SOURCE)\n\n"
            f"```json\n{extraction_text}\n```\n\n"
            f"## Question to Answer (eventually)\n"
            f"<q>{question}</q>\n\n"
            "Your task: Based ONLY on the extracted image content above, design a "
            "data-collection plan (steps) that an executor can follow to find the answer. "
            "Remember: you cannot see the image — you only have the JSON above. "
            "Output ONLY the JSON plan."
        )

        messages = [{'type': 'text', 'value': user_prompt}]

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = BLIND_PLANNING_PROMPT
        try:
            ret, text, resp = self._api_call(
                messages, schema=PLANNING_SCHEMA, with_image_processor=False
            )
        finally:
            self.agent.system_prompt = saved_sp

        plan_obj, err = self._parse_json(text)
        if plan_obj is None or 'steps' not in plan_obj:
            log(f"[Stage 1] FAILED: {err}")
            return None, 'planning_failed', text or ''

        steps = plan_obj['steps']
        seen_ids = set()
        for s in steps:
            sid = s.get('id', '')
            if not sid or sid in seen_ids:
                log(f"[Stage 1] Invalid step id: {sid}")
                return None, 'planning_failed', text or ''
            seen_ids.add(sid)
            for dep in s.get('depends_on', []):
                if dep not in seen_ids:
                    log(f"[Stage 1] Step {sid} depends on unseen {dep}")
                    return None, 'planning_failed', text or ''

        log(f"[Stage 1] OK — {len(steps)} steps:")
        for s in steps:
            log(f"  {s['id']} ({s['type']}) deps={s.get('depends_on', [])}: {s['description'][:100]}")
        return steps, 'success', text

    # ==================================================================
    # Stage 2: Executor (same as V4)
    # ==================================================================
    def stage2_executor(self, steps, image_path, orig_w, orig_h):
        """执行所有step（拓扑序）"""
        log("=" * 60)
        log(f"[Stage 2] Executor — {len(steps)} steps")
        log("=" * 60)

        all_results = {}
        overall_status = 'success'

        for step in steps:
            step_id = step.get('id', 'unknown')
            step_type = step.get('type', 'unknown')
            log(f"\n--- {step_id} ({step_type}) ---")

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
                        'status': 'error', 'type': step_type,
                        'error': f'Unknown type: {step_type}', 'result': ''
                    }
            except Exception as e:
                result_data = {
                    'status': 'error', 'type': step_type,
                    'error': str(e), 'result': ''
                }

            all_results[step_id] = result_data
            status = result_data.get('status', '?')
            preview = str(result_data.get('result', ''))[:120]
            log(f"  → {status}: {preview}")

            if result_data.get('status') == 'error':
                if step_type == 'ground-understand':
                    if overall_status == 'success':
                        overall_status = 'grounding_error'
                else:
                    if overall_status == 'success':
                        overall_status = 'execution_error'

        return all_results, overall_status

    def _execute_ground_understand(self, step, context, image_path, orig_w, orig_h):
        description = step.get('description', '')
        step_id = step.get('id', 'unknown')

        # Phase 1: Locate
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
                locate_messages, schema=LOCATE_SCHEMA, with_image_processor=True
            )
        finally:
            self.agent.system_prompt = saved_sp

        locate_obj, locate_err = self._parse_json(locate_text)
        log(f"  [Locate] err={locate_err}")

        if locate_obj is None or not locate_obj.get('bbox_groups'):
            return {
                'status': 'error', 'type': 'ground-understand', 'result': '',
                'error': f'Locate failed: {locate_err or "empty bbox_groups"}'
            }

        bbox_groups = locate_obj['bbox_groups']
        self._draw_bboxes(image_path, bbox_groups, step_id, orig_w, orig_h)

        # Collect crops
        all_crops = []
        for group in bbox_groups:
            group_id = group.get('group_id', 'unknown')
            purpose = group.get('purpose', '')
            for i, bbox in enumerate(group.get('bboxes', [])):
                if len(bbox) != 4: continue
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(1000, int(x1)))
                y1 = max(0, min(1000, int(y1)))
                x2 = max(0, min(1000, int(x2)))
                y2 = max(0, min(1000, int(y2)))
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1
                if x1 == x2 or y1 == y2: continue
                px1 = int(x1 / 1000.0 * orig_w)
                py1 = int(y1 / 1000.0 * orig_h)
                px2 = int(x2 / 1000.0 * orig_w)
                py2 = int(y2 / 1000.0 * orig_h)
                all_crops.append((group_id, purpose, i, (px1, py1, px2, py2)))

        if not all_crops:
            return {
                'status': 'error', 'type': 'ground-understand', 'result': '',
                'error': 'No valid bboxes', 'bbox_groups': bbox_groups
            }

        # Phase 2: Understand
        understand_content = [
            {'type': 'text', 'value': (
                f"Task: {description}\n\n{context}\n"
                "Below are cropped image regions. Read and synthesize. Output ONLY JSON."
            )}
        ]
        current_group = None
        for group_id, purpose, bbox_idx, phys_bbox in all_crops:
            if group_id != current_group:
                current_group = group_id
                understand_content.append({
                    'type': 'text',
                    'value': f'\n--- Region Group: {group_id} (Purpose: {purpose}) ---'
                })
            understand_content.append({
                'type': 'text',
                'value': f'[Group {group_id}, bbox {bbox_idx + 1}]',
            })
            b64 = self._crop_and_encode(image_path, phys_bbox)
            understand_content.append({'type': 'image_base64', 'value': b64})

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = UNDERSTAND_PROMPT
        try:
            ret, understand_text, _ = self._api_call(
                understand_content, schema=UNDERSTAND_SCHEMA, with_image_processor=False
            )
        finally:
            self.agent.system_prompt = saved_sp

        understand_obj, understand_err = self._parse_json(understand_text)
        log(f"  [Understand] err={understand_err}")

        if understand_obj is None:
            return {
                'status': 'error', 'type': 'ground-understand', 'result': '',
                'error': f'Understand failed: {understand_err}', 'bbox_groups': bbox_groups
            }

        return {
            'status': 'ok', 'type': 'ground-understand',
            'result': understand_obj.get('result', ''), 'bbox_groups': bbox_groups
        }

    def _execute_compute(self, step, context):
        description = step.get('description', '')
        user_prompt = (
            f"Task: {description}\n\n{context}\n"
            "Perform the calculation. Output ONLY the JSON result."
        )
        messages = [{'type': 'text', 'value': user_prompt}]
        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = COMPUTE_PROMPT
        try:
            ret, text, _ = self._api_call(messages, schema=COMPUTE_SCHEMA, with_image_processor=False)
        finally:
            self.agent.system_prompt = saved_sp
        obj, err = self._parse_json(text)
        if obj is None:
            return {'status': 'error', 'type': 'compute', 'result': '', 'error': f'Compute failed: {err}'}
        return {'status': 'ok', 'type': 'compute', 'result': obj.get('result', '')}

    def _execute_logical(self, step, context):
        description = step.get('description', '')
        user_prompt = (
            f"Task: {description}\n\n{context}\n"
            "Apply the logical condition. Output ONLY the JSON result."
        )
        messages = [{'type': 'text', 'value': user_prompt}]
        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = LOGICAL_PROMPT
        try:
            ret, text, _ = self._api_call(messages, schema=LOGICAL_SCHEMA, with_image_processor=False)
        finally:
            self.agent.system_prompt = saved_sp
        obj, err = self._parse_json(text)
        if obj is None:
            return {'status': 'error', 'type': 'logical', 'result': '', 'error': f'Logical failed: {err}'}
        return {'status': 'ok', 'type': 'logical', 'result': obj.get('result', '')}

    # ==================================================================
    # Stage 3: Reasoning (text-only)
    # ==================================================================
    def stage3_reasoning(self, question, all_step_results):
        """最终推理：纯文本，无图像"""
        log("=" * 60)
        log("[Stage 3] Reasoning — text-only, no image")
        log("=" * 60)

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
        user_content = [
            {'type': 'text', 'value': (
                f"Original question: <q>{question}</q>\n\n"
                "You have NO image. Use ONLY the analysis results above. "
                "If data is missing due to FAILED steps, say so. "
                "Enclose your final answer in <a> and </a> tags."
            )},
        ]

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = reasoning_prompt
        try:
            ret, answer, resp = self._api_call(
                user_content, schema=None, with_image_processor=False
            )
        finally:
            self.agent.system_prompt = saved_sp

        log(f"[Stage 3] ret={ret}, answer_len={len(answer) if answer else 0}")
        return ret, answer, resp

    # ==================================================================
    # Fallback: direct image reasoning
    # ==================================================================
    def fallback_reasoning(self, image_path, question):
        log("[Fallback] Direct image reasoning")
        prompt = (
            f"Question: <q>{question}</q>\n\n"
            "Look at the image carefully and answer step by step. "
            "Enclose your final answer in <a> and </a> tags."
        )
        messages = [
            {'type': 'text', 'value': prompt},
            {'type': 'image', 'value': image_path},
        ]
        ret, answer, resp = self._api_call(messages, schema=None, with_image_processor=True)
        return ret, answer, resp

    # ==================================================================
    # Main: Blind Pipeline
    # ==================================================================
    def run_blind_pipeline(self, image_path, question):
        """完整的 Blind Pipeline"""
        log("\n" + "█" * 60)
        log("█  BLIND PLANNING PIPELINE")
        log("█" * 60)

        orig_w, orig_h = Image.open(image_path).size

        # Stage 0: Visual Extraction
        t0 = _time.time()
        extraction, extraction_raw = self.stage0_extraction(image_path)
        t0_e = _time.time() - t0

        if extraction is None:
            log("[BLIND] Extraction failed — falling back to direct reasoning")
            ret, answer, resp = self.fallback_reasoning(image_path, question)
            return {
                'pipeline': 'blind',
                'extraction': None,
                'plan': None,
                'exec_results': {},
                'answer': answer,
                'ret': ret,
                'status': 'extraction_failed',
                'timing': {'extraction': t0_e},
            }

        # Stage 1: Blind Planning (text-only)
        t1 = _time.time()
        steps, plan_status, plan_text = self.stage1_blind_planning(extraction, question)
        t1_e = _time.time() - t1

        # Stage 2: Executor
        if plan_status == 'success' and steps:
            t2 = _time.time()
            all_results, exec_status = self.stage2_executor(steps, image_path, orig_w, orig_h)
            t2_e = _time.time() - t2
        else:
            all_results = {}
            exec_status = plan_status
            t2_e = 0

        # Stage 3: Reasoning
        t3 = _time.time()
        if plan_status == 'success' and all_results:
            ret, answer, resp = self.stage3_reasoning(question, all_results)
        else:
            ret, answer, resp = self.fallback_reasoning(image_path, question)
        t3_e = _time.time() - t3

        total = _time.time() - t0

        # Status
        if extraction is None:
            final_status = 'extraction_failed'
        elif plan_status == 'planning_failed':
            final_status = 'planning_failed'
        elif exec_status != 'success':
            final_status = exec_status
        else:
            final_status = 'success'

        n_ok = sum(1 for r in all_results.values() if r.get('status') == 'ok')
        n_err = len(steps or []) - n_ok

        log(f"\n[BLIND] Done. status={final_status} | steps={len(steps or [])} ok={n_ok} err={n_err}")
        log(f"[BLIND] Timing: Extraction={t0_e:.0f}s Planning={t1_e:.0f}s Exec={t2_e:.0f}s Reasoning={t3_e:.0f}s Total={total:.0f}s")

        return {
            'pipeline': 'blind',
            'extraction': extraction,
            'plan': {'steps': steps, 'status': plan_status, 'raw': plan_text},
            'exec_results': all_results,
            'answer': answer,
            'ret': ret,
            'status': final_status,
            'timing': {
                'extraction': t0_e, 'planning': t1_e,
                'executor': t2_e, 'reasoning': t3_e, 'total': total
            },
        }

    # ==================================================================
    # Main: Original V4 Pipeline (for comparison)
    # ==================================================================
    def run_original_v4(self, image_path, question):
        """运行原始 V4 agent 做对比"""
        log("\n" + "█" * 60)
        log("█  ORIGINAL V4 PIPELINE (for comparison)")
        log("█" * 60)

        t_start = _time.time()

        agent = self.v4_agent_ctor(**self.common_kwargs)

        # Monkey-patch to capture internal state
        captured = {'plan': None, 'exec_results': None}

        orig_generate_inner = agent.generate_inner

        def wrapper(inputs, **kwargs):
            # Extract image & question
            image_path_inner = None
            question_inner = ""
            for item in inputs:
                if item['type'] == 'image':
                    image_path_inner = item['value']
                elif item['type'] == 'text':
                    question_inner += item['value']

            if not image_path_inner:
                return orig_generate_inner(inputs, **kwargs)

            with Image.open(image_path_inner) as img:
                ow, oh = img.size

            # Stage 1
            steps, ps, pt = agent._planning(image_path_inner, question_inner)
            captured['plan'] = {'steps': steps, 'status': ps, 'raw': pt}

            # Stage 2
            if ps == 'success' and steps:
                all_res, es = agent._execute_steps(steps, image_path_inner, ow, oh)
                captured['exec_results'] = all_res
            else:
                all_res = {}
                es = ps

            # Stage 3
            if ps == 'success' and all_res:
                ret, answer, resp = agent._reasoning(image_path_inner, question_inner, all_res)
            else:
                ret, answer, resp = agent._fallback_reasoning(image_path_inner, question_inner)

            return ret, answer, resp

        agent.generate_inner = wrapper

        inputs = [
            {'type': 'text', 'value': question},
            {'type': 'image', 'value': image_path},
        ]
        ret, answer, resp = agent.generate_inner(inputs)

        total = _time.time() - t_start
        log(f"[ORIG V4] Done. Total={total:.0f}s")

        return {
            'pipeline': 'original_v4',
            'plan': captured['plan'],
            'exec_results': captured['exec_results'],
            'answer': answer,
            'ret': ret,
            'status': 'unknown',
            'timing': {'total': total},
        }


# ============================================================================
# 对比分析
# ============================================================================
def compare_plans(blind_result, original_result):
    """对比两个 pipeline 的 plan"""
    print("\n" + "=" * 70)
    print("║  PLAN COMPARISON: Blind vs Original V4")
    print("=" * 70)

    blind_plan = blind_result.get('plan') or {}
    orig_plan = original_result.get('plan') or {}

    blind_steps = blind_plan.get('steps') or []
    orig_steps = orig_plan.get('steps') or []

    print(f"\n{' Blind Planning (JSON→DAG) ':=^60}")
    if blind_steps:
        for s in blind_steps:
            print(f"  {s['id']} | {s['type']:20s} | deps={s.get('depends_on', [])}")
            print(f"       └ {s['description'][:120]}")
    else:
        print(f"  (failed: {blind_plan.get('status', '?')})")

    print(f"\n{' Original V4 (Image→DAG) ':=^60}")
    if orig_steps:
        for s in orig_steps:
            print(f"  {s['id']} | {s['type']:20s} | deps={s.get('depends_on', [])}")
            print(f"       └ {s['description'][:120]}")
    else:
        print(f"  (failed: {orig_plan.get('status', '?')})")

    print(f"\n{' Answers ':=^60}")
    print(f"  Blind:     {blind_result.get('answer', 'N/A')[:200]}")
    print(f"  Original:  {original_result.get('answer', 'N/A')[:200]}")


# ============================================================================
# Main
# ============================================================================
def main():
    log("=" * 60)
    log("Blind Planning Test: 验证 Planning 是否数据驱动")
    log("=" * 60)
    log(f"Image:  {LOCAL_IMAGE_PATH}")
    log(f"Question: {QUESTION}")

    if not os.path.exists(LOCAL_IMAGE_PATH):
        log(f"图片不存在: {LOCAL_IMAGE_PATH}")
        sys.exit(1)

    pipeline = BlindPlanningPipeline()

    # 1. 运行 Blind Pipeline
    blind_result = pipeline.run_blind_pipeline(LOCAL_IMAGE_PATH, QUESTION)

    # 2. 运行原始 V4 (对比)
    original_result = pipeline.run_original_v4(LOCAL_IMAGE_PATH, QUESTION)

    # 3. 对比
    compare_plans(blind_result, original_result)

    # 4. 保存结果
    result = {
        "image_path": LOCAL_IMAGE_PATH,
        "question": QUESTION,
        "blind_pipeline": {
            "extraction": blind_result.get('extraction'),
            "plan": {
                "steps": blind_result.get('plan', {}).get('steps'),
                "status": blind_result.get('plan', {}).get('status'),
            },
            "exec_results": {
                k: {'status': v.get('status'), 'result': str(v.get('result', ''))[:200]}
                for k, v in (blind_result.get('exec_results') or {}).items()
            },
            "answer": blind_result.get('answer'),
            "status": blind_result.get('status'),
            "timing": blind_result.get('timing'),
        },
        "original_v4": {
            "plan": {
                "steps": original_result.get('plan', {}).get('steps'),
                "status": original_result.get('plan', {}).get('status'),
            },
            "exec_results": {
                k: {'status': v.get('status'), 'result': str(v.get('result', ''))[:200]}
                for k, v in (original_result.get('exec_results') or {}).items()
            },
            "answer": original_result.get('answer'),
            "status": original_result.get('status'),
            "timing": original_result.get('timing'),
        },
    }

    result_path = os.path.join(OUTPUT_DIR, "blind_planning_comparison.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"\n结果已保存: {result_path}")

    # 也保存 extraction 供人工检查
    if blind_result.get('extraction'):
        ext_path = os.path.join(OUTPUT_DIR, "extraction_output.json")
        with open(ext_path, 'w', encoding='utf-8') as f:
            json.dump(blind_result['extraction'], f, ensure_ascii=False, indent=2)
        log(f"Extraction 已保存: {ext_path}")


if __name__ == '__main__':
    main()
