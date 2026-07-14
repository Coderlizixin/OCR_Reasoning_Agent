"""
OCR-Reasoning Agent v3 — Sequential Grounding → Understanding Pipeline.

Architecture:
  Phase 1: Structured Planning (JSON schema, atomic steps, DAG with $placeholders)
  Phase 2: Sequential Execution Loop
    - <ground-understand>: Grounding (multi-bbox) → Understanding (indexed values)
    - <compute>: LLM computation
    - <logical>: LLM for filtering/judgment
    - $step_id placeholders resolved at runtime from dependency evidence
  Phase 3: Final Reasoning with complete evidence chain

Key design decisions:
  - NO re-planning — failed steps stay FAILED (simpler, faster, more predictable)
  - FALLBACK: if planning fails, direct image+question reasoning
  - Grounding and Understanding are SEPARATED (2 API calls per visual step)
  - Multiple bboxes allowed per step (e.g., legend + axis + data point for one chart reading)
  - Dependency context is PURE TEXT (no crop images passed between steps)
  - Understanding outputs indexed values: [1]=val1, [2]=val2, ...
  - Placeholder mechanism: $s2 in description → resolved at runtime
  - target_desc REMOVED — description now serves both semantic AND spatial roles
  - Crop limit (MAX_CROPS=44) with overflow as text-only region descriptions
"""

import os
import io
import re
import json
import base64
import time as _time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from vlmeval.api.gpt import OpenAIWrapper, _log_diag


def clean_value(value_str: str):
    """Strip currency symbols, whitespace, and noise. Returns (cleaned, was_cleaned)."""
    if not value_str:
        return None, False
    original = value_str
    for sym in ['$', '€', '¥', '£', '￥', '₹', '%', '°']:
        value_str = value_str.replace(sym, '')
    value_str = re.sub(r'\s*\(.*?\)\s*$', '', value_str)
    value_str = value_str.strip().rstrip('.,;:')
    was_cleaned = (value_str != original.strip())
    if not value_str:
        return None, True
    try:
        if '.' in value_str:
            return float(value_str), was_cleaned
        else:
            return int(value_str), was_cleaned
    except ValueError:
        return value_str, was_cleaned


def extract_final_value(text: str) -> Optional[str]:
    """Extract value from verbose model output (handles thinking traces)."""
    if not text:
        return None

    for tag in ['<think>', ' response', ' response']:
        parts = text.split(tag)
        if len(parts) > 1:
            after = parts[-1]
            for end_tag in ['</think>', '</think>', '</think>']:
                if end_tag in after:
                    after = after.rsplit(end_tag, 1)[0]
            after = after.strip()
            if after:
                text = after
                break

    text = re.sub(r'^/no_think\s*', '', text)
    text = re.sub(r'^/think\s*', '', text)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None

    short_lines = [l for l in lines if len(l) < 200]
    if short_lines:
        text = short_lines[-1]
    else:
        text = lines[-1]

    text = re.sub(r'\s*\(conf[^)]*\)\s*$', '', text)
    text = text.strip()
    return text


# ============================================================
# Main Agent Class
# ============================================================

class OCRReasoningAgentV3(OpenAIWrapper):
    """OCR-Reasoning Agent V3 with sequential grounding→understanding pipeline.

    Extends OpenAIWrapper for full compatibility with the VLMEvalKit framework.
    Overrides generate_inner() to implement the 3-phase agent pipeline
    (Phase 1: Planning → Phase 2: Execute → Phase 3: Reason).
    If planning fails, falls back to direct image+question reasoning.
    """

    def __init__(self,
                 padding: int = 50,
                 scale_factor: float = 2.5,
                 max_crop_edge: Optional[int] = None,
                 max_crops: int = 44,
                 max_iterations: int = 30,
                 planner_model: str = None,
                 planner_api_base: str = None,
                 grounder_model: str = None,
                 grounder_api_base: str = None,
                 **kwargs):
        """
        Args:
            padding: pixel padding for tight crops
            scale_factor: magnification for tight crops
            max_crop_edge: max edge length (px) for crop images (None = no limit)
            max_crops: maximum number of cropped sub-images per step (overflow → text-only)
            max_iterations: max iterations for the sequential execution loop
            planner_model: override model name for planning (defaults to parent's model)
            planner_api_base: override API base for planning
            grounder_model: override model name for grounding
            grounder_api_base: override API base for grounding
        """
        super().__init__(**kwargs)
        self.padding = padding
        self.scale_factor = scale_factor
        self.max_crop_edge = max_crop_edge
        self.max_crops = max_crops
        self.max_iterations = max_iterations

        if 'max_tokens' not in kwargs:
            self.max_tokens = 6144

        # ---- Allow thinking mode for all instances ----
        if self.extra_body is None:
            self.extra_body = {}

        # ---- Planner instance (Phase 1) ----
        planner_kwargs = dict(
            model=planner_model or self.model,
            api_base=planner_api_base or self.api_base,
            key=getattr(self, 'key', 'sk-EMPTY'),
            temperature=0.0,
            max_tokens=12288,
            img_size=self.img_size,
            img_detail=self.img_detail,
            extra_body=self.extra_body,
            verbose=self.verbose,
        )
        self.planner = OpenAIWrapper(**planner_kwargs)

        # ---- Grounder instance (Phase 2: grounding) ----
        # Uses moderate resolution for grounding (different from full-res planning)
        grounder_extra_body = {
            "mm_processor_kwargs": {
                "images_kwargs": {
                    "size": {
                        "longest_edge": 2048 * 32 * 32,
                        "shortest_edge": 64 * 32 * 32,
                    }
                }
            }
        }
        grounder_kwargs = dict(
            model=grounder_model or self.model,
            api_base=grounder_api_base or self.api_base,
            key=getattr(self, 'key', 'sk-EMPTY'),
            temperature=0.0,
            max_tokens=6144,
            img_size=self.img_size,
            img_detail=self.img_detail,
            extra_body=grounder_extra_body,
            verbose=self.verbose,
        )
        self.grounder = OpenAIWrapper(**grounder_kwargs)

    # ============================================================
    # Image Utilities (in-memory crop + encode)
    # ============================================================

    def _encode_full_resolution(self, image_path: str) -> Tuple[str, int, int]:
        """Read image at FULL resolution — no scaling, no compression.
        Used for Planning and Grounding phases."""
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size
            buffer = io.BytesIO()
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(buffer, format="JPEG", quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8'), orig_w, orig_h

    def _resize_and_encode_original(self, image_path: str, max_edge: int = 1024) -> str:
        """Encode original image scaled down to max_edge for global context."""
        with Image.open(image_path).convert('RGB') as img:
            w, h = img.size
            _max = max(w, h)
            if _max > max_edge:
                ratio = max_edge / _max
                img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _crop_and_encode_base64(self, image_path: str, physical_bbox: List[int]) -> str:
        """Crop at physical coordinates, magnify, encode to base64."""
        with Image.open(image_path).convert('RGB') as img:
            xmin, ymin, xmax, ymax = physical_bbox
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            if xmin == xmax:
                xmax += 1
            if ymin == ymax:
                ymax += 1

            c_xmin = max(0, xmin - self.padding)
            c_ymin = max(0, ymin - self.padding)
            c_xmax = min(img.width, xmax + self.padding)
            c_ymax = min(img.height, ymax + self.padding)

            cropped = img.crop((c_xmin, c_ymin, c_xmax, c_ymax))

            if self.scale_factor != 1.0:
                new_w = int(cropped.width * self.scale_factor)
                new_h = int(cropped.height * self.scale_factor)
                cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

            if self.max_crop_edge is not None:
                w, h = cropped.size
                _max = max(w, h)
                if _max > self.max_crop_edge:
                    ratio = self.max_crop_edge / _max
                    cropped = cropped.resize((int(w * ratio), int(h * ratio)),
                                             Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            cropped.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _encode_pil_to_base64(self, pil_image: Image.Image, max_edge: int = 1024) -> str:
        """Encode PIL image to base64 JPEG with max edge constraint."""
        w, h = pil_image.size
        _max = max(w, h)
        if max_edge is not None and _max > max_edge:
            ratio = max_edge / _max
            pil_image = pil_image.resize((int(w * ratio), int(h * ratio)),
                                         Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        if pil_image.mode in ("RGBA", "P"):
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    # ============================================================
    # Message Construction Helpers
    # ============================================================

    def prepare_itlist(self, inputs):
        """Extend parent to support 'image_base64' type for in-memory crops."""
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

    # ============================================================
    # JSON Parsing (Robust)
    # ============================================================

    @staticmethod
    def _parse_json_robust(text: str) -> Optional[Dict]:
        """Robustly extract JSON from model output (handles thinking traces)."""
        if not text:
            return None

        for tag in ['<think>', ' response', ' response']:
            for end_tag in ['</think>', '</think>', '</think>']:
                if end_tag in text:
                    parts = text.rsplit(end_tag, 1)
                    text = parts[-1]
                if tag in text and end_tag in text:
                    text = re.sub(re.escape(tag) + r'.*?' + re.escape(end_tag),
                                 '', text, flags=re.DOTALL)

        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = re.sub(r'^/no_think\s*', '', text)
        text = re.sub(r'^/think\s*', '', text)

        for attempt in range(5):
            try:
                return json.loads(text.strip())
            except json.JSONDecodeError:
                pass
            if attempt == 0:
                start = text.find('{')
                end = text.rfind('}')
                if start >= 0 and end > start:
                    text = text[start:end + 1]
            elif attempt == 1:
                text = re.sub(r',\s*}', '}', text)
                text = re.sub(r',\s*]', ']', text)
            elif attempt == 2:
                lines = text.split('\n')
                json_lines = [l for l in lines
                              if re.search(r'[{}\[\]:"]', l)
                              or l.strip().startswith(('{', '}', '[', ']'))]
                text = '\n'.join(json_lines)
            elif attempt == 3:
                text = re.sub(r'(\s+)(\w+)(\s*):', r'\1"\2"\3:', text)
        return None

    # ============================================================
    # Placeholder Resolution
    # ============================================================

    @staticmethod
    def _resolve_placeholders(text: str, evidence: Dict[str, Dict]) -> str:
        """Replace $step_id placeholders with actual values from evidence.

        Supports:
          $s2       → the value of step s2
          $s2.xxx   → sub-field
        """
        if not text:
            return text

        def replace_ref(match):
            ref = match.group(1)
            parts = ref.split('.', 1)
            step_id = parts[0]

            ev = evidence.get(step_id, {})
            if not ev:
                return f"[UNRESOLVED:{ref}]"

            if len(parts) == 1:
                val = ev.get('value')
                if val is None:
                    return f"[UNRESOLVED:{ref}]"
                if isinstance(val, (list, dict)):
                    return json.dumps(val, ensure_ascii=False)
                return str(val)
            else:
                sub = parts[1]
                sub_fields = ev.get('sub_fields', {})
                if sub in sub_fields:
                    return str(sub_fields[sub])
                return f"[UNRESOLVED:{ref}]"

        return re.sub(r'\$(\w+(?:\.\w+)*)', replace_ref, text)

    # ============================================================
    # Phase 1: Structured Planning
    # ============================================================

    def _build_planning_prompt(self, question: str) -> str:
        """Build the V3 planning prompt — enforces fine-grained atomic steps."""
        return (
            f"Create a JSON execution plan to answer this question: <q>{question}</q>\n\n"
            "=== ROLE ===\n"
            "You are a PLANNER. Decompose the question into SMALL, ATOMIC steps.\n"
            "Each step does ONE tiny thing. Output ONLY the JSON plan.\n\n"
            "=== STEP TYPES ===\n"
            '- "<ground-understand>": Read ONE kind of element from ONE contiguous '
            'region and understand its meaning. Needs image.\n'
            '- "<compute>": ONE arithmetic operation (sum, avg, diff, etc.). '
            'No image.\n'
            '- "<logical>": ONE filter, sort, group, or match operation. '
            'No image.\n\n'
            "=== WHAT TO SKIP (IMPORTANT) ===\n"
            "Do NOT create steps for:\n"
            "- Reading the chart title, subtitle, or captions (irrelevant metadata)\n"
            "- Reading the legend (unless the question asks about categories "
            "shown IN the legend)\n"
            "- Describing the overall image layout\n"
            "Only create steps that DIRECTLY contribute to answering the question.\n\n"
            "=== WHAT MAKES A STEP TRULY ATOMIC? ===\n"
            "WHAT ARE ATOMIC:\n"
            '"all item names in the left column"\n'
            '"the value next to label X"\n'
            '"the blue section of the pie chart"\n'
            '"the y-axis scale numbers"\n'
            '"column headers in the top row"\n'
            'WHAT ARE NOT ATOMIC (✗):\n'
            '"all item names AND their prices"\n'
            '"all data in the table"\n'
            '"the entire chart"\n'
            '"the axes and data series"\n'
            '"the headers and first 5 rows"\n\n'
            "=== GOLDEN RULES ===\n"
            "R1. FILTERING IS A SEPARATE logical STEP: Never put conditions "
            "into <ground-understand>.\n"
            "R2. ONE SEMANTIC UNIT PER STEP: One column, one row, one axis, "
            "one label group, one value. If a description contains \"and\" or "
            "\"with their\", SPLIT IT.\n"
            "R3. USE $step_id PLACEHOLDERS: \"filter $s1 where price > $s2\".\n\n"
            "=== CONCRETE EXAMPLE ===\n"
            'Question: "For the countries whose steel export volume is higher '
            'than the average, what is the total amount of their oil imports?"\n'
            "Structure: type=chart, 2 bars [steel export, oil import] for each "
            "country, 5 countries\n\n"
            "GOOD plan (5 steps):\n"
            '  s1 (<ground-understand>): "Read steel export of all countries."\n'
            '  s2 (<compute>): "Compute average of steel export."\n'
            '  s3 (<logical>): "Select the countries whose steel export are '
            'higher than the average $s2."\n'
            '  s4 (<ground-understand>): "Read oil import of the countries '
            'in $s3."\n'
            '  s5 (<compute>): "Compute the total amount of oil imports in '
            'countries $s4."\n'
            "BAD plan (3 steps — TOO COARSE):\n"
            '  s1: "Read all employee data from the chart."\n'
            '  s2: "Select the countries whose steel export are higher than '
            'the average."\n'
            '  s3: "Compute the total amount of oil imports in countries."\n\n'
            "=== JSON SCHEMA ===\n"
            '{"steps": [\n'
            '  {"id": "s1", "type": "<ground-understand>", '
            '"description": "Read all X from region Y. Output as list.", '
            '"expected_type": "list[str]", "depends_on": []},\n'
            '  ...\n'
            ']}\n\n'
            "OUTPUT ONLY THE JSON. No thinking, no markdown, no explanation."
        )

    def structured_planning(self, image_path: str, question: str) -> Tuple[List[Dict], str]:
        """Phase 1: Generate a structured execution plan from the image + question.

        Returns:
            (plan_steps, raw_plan_text) — list of step dicts and raw model output.
        """
        _t_start = _time.time()
        _log_diag('[V3 Phase1] Structured Planning start')

        prompt = self._build_planning_prompt(question)

        plan_inputs = [
            {'type': 'text', 'value': prompt},
            {'type': 'image', 'value': image_path},
        ]

        ret, plan_text, resp = self.planner.generate_inner(plan_inputs)
        _t_elapsed = _time.time() - _t_start

        if ret != 0:
            _log_diag(f'[V3 Phase1] FAILED after {_t_elapsed:.0f}s')
            return [], plan_text

        plan_json = self._parse_json_robust(plan_text)
        if plan_json is None:
            _log_diag(f'[V3 Phase1] JSON parse failed. Raw: {plan_text[:300]}...')
            return [], plan_text

        # Parse into step dicts (no target_desc — removed)
        plan_steps = []
        raw_steps = plan_json.get('steps', [])
        for i, raw in enumerate(raw_steps):
            step = {
                'id': raw.get('id', f's{i+1}'),
                'type': raw.get('type', '<ground-understand>'),
                'description': raw.get('description', ''),
                'expected_type': raw.get('expected_type', 'text'),
                'expression': raw.get('expression', ''),
                'depends_on': raw.get('depends_on', []),
                'priority': raw.get('priority', i + 1),
            }
            plan_steps.append(step)

        _log_diag(f'[V3 Phase1] Done in {_t_elapsed:.0f}s | {len(plan_steps)} steps')
        return plan_steps, plan_text

    # ============================================================
    # Phase 2: Sequential Execution Loop
    # ============================================================

    def _ground_step(self, step: Dict, image_path: str, orig_w: int, orig_h: int,
                     dep_context: str, evidence: Dict[str, Dict],
                     ) -> Tuple[List[List[int]], List[str]]:
        """Grounding: locate visual elements for a single step.

        Returns (valid_bboxes, valid_labels).
        """
        _t_start = _time.time()
        _log_diag(f'[V3 Grounding] {step["id"]}')

        resolved_description = self._resolve_placeholders(
            step.get('description', ''), evidence)

        grounding_prompt = (
            "You are a visual grounding assistant. Locate visual elements with "
            "SEMANTIC labels AND coordinates. Besides, give the answer of your "
            "target.\n\n"
            f"=== CONTEXT FROM PREVIOUS STEPS ===\n{dep_context}\n\n"
            f"=== TASK ===\n"
            f"Target: {resolved_description}\n\n"
            "=== CRITICAL: OUTPUT FORMAT (MANDATORY) ===\n"
            "Each element MUST have a descriptive label. "
            "Use the format EXACTLY as shown:\n"
            '  "semantic description" -> [xmin, ymin, xmax, ymax]\n\n'
            "RIGHT:\n"
            '  "the price of Burger" -> [350, 200, 420, 230]\n'
            '  "column header: Salary" -> [500, 50, 620, 80]\n'
            '  "legend: Very" -> [420, 650, 440, 660]\n\n'
            "WRONG (bare coords):\n"
            "  [350, 200, 420, 230]\n\n"
            "=== OUTPUT FORMAT ===\n"
            "Step 1 — Output coordinate lines in this EXACT format:\n"
            '  "semantic description" -> [xmin, ymin, xmax, ymax]\n'
            "  Coordinates are IMAGE PIXEL POSITIONS normalized to [0,1000].\n"
            "  (0,0)=top-left, (1000,1000)=bottom-right. All values integers.\n"
            '  Example: "the price label next to Burger" -> [350, 200, 420, 230]\n\n'
            "Step 2 — On a NEW LINE after all coordinates, answer the target "
            "question in one sentence, prefixed with 'ANSWER: '.\n"
            '  Example: "ANSWER: The price of Burger is $2."\n\n'
            "=== RULES ===\n"
            "1. MANDATORY: Every line must have a semantic label + arrow + "
            "coords.\n"
            "2. Box the TIGHTEST rectangle. Normalize to [0,1000] where (0,0) "
            "is top-left and (1000,1000) is bottom-right. All values must be "
            "integers..\n"
            "3. Each box = ONE element. Don't group unrelated items.\n"
            "4. If a requested NUMBER or TEXT is not printed inside an element,\n"
            "   box the element itself (the bar, the cell, the region) and\n"
            "   label it clearly. NEVER output text-only explanations without\n"
            "   coordinates.\n"
            "Output ONLY labeled coordinate lines. And give the answer of your "
            "target, e.g., 'The price of water is $2.' 'The legent Very is "
            "colored in dark gold.'\n"
        )

        ground_inputs = [
            {'type': 'text', 'value': grounding_prompt},
            {'type': 'image', 'value': image_path},
        ]

        ret, grounding_text, resp = self.grounder.generate_inner(ground_inputs)
        _t_elapsed = _time.time() - _t_start
        _log_diag(f'[V3 Grounding] {step["id"]} done in {_t_elapsed:.0f}s | '
                  f'{len(grounding_text) if grounding_text else 0} chars')

        if ret != 0 or not grounding_text:
            return [], []

        # Parse labeled coordinates (accept both Unicode → and ASCII ->)
        labeled_pat = (r'"([^"]*)"\s*(?:→|->)\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,'
                       r'\s*(-?\d+)\s*,\s*(-?\d+)\s*\]')
        matches = re.findall(labeled_pat, grounding_text)

        if not matches:
            # Try bare coordinates
            bare_pat = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
            bare_matches = re.findall(bare_pat, grounding_text)
            if bare_matches:
                matches = [(f"Region {i+1}",) + bm
                          for i, bm in enumerate(bare_matches)]
                _log_diag(f'[V3 Grounding] {step["id"]}: parsed {len(matches)} '
                          f'bare coords')

        if not matches:
            _log_diag(f'[V3 Grounding] {step["id"]}: no coordinates found')
            return [], []

        # Validate bboxes
        valid_bboxes = []
        valid_labels = []
        for m in matches:
            raw_label = m[0].strip()
            bbox = [int(m[1]), int(m[2]), int(m[3]), int(m[4])]

            # Null check
            if bbox[0] < 0 or bbox[1] < 0:
                continue
            if bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0:
                continue
            # Range check
            if not (0 <= bbox[0] <= 1000 and 0 <= bbox[1] <= 1000 and
                    0 <= bbox[2] <= 1000 and 0 <= bbox[3] <= 1000):
                continue
            # Positive area check
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            # Aspect ratio sanity check
            w_norm = bbox[2] - bbox[0]
            h_norm = bbox[3] - bbox[1]
            if h_norm > 0 and w_norm > 0:
                aspect = max(w_norm / h_norm, h_norm / w_norm)
                if aspect > 50:
                    _log_diag(f'[V3 Grounding] Skipping extreme aspect '
                              f'{aspect:.1f}: {bbox}')
                    continue

            valid_bboxes.append(bbox)
            valid_labels.append(raw_label)

        _log_diag(f'[V3 Grounding] {step["id"]}: {len(valid_bboxes)} valid bboxes')
        return valid_bboxes, valid_labels

    def _understand_step(self, step: Dict, image_path: str, orig_w: int, orig_h: int,
                         cropped_bboxes: List[List[int]], cropped_labels: List[str],
                         overflow_bboxes: List[List[int]], overflow_labels: List[str],
                         dep_context: str, evidence: Dict[str, Dict]) -> Optional[str]:
        """Understanding: answer question using crops + original image.

        Returns answer text with indexed values: [1]=val1, [2]=val2, ...
        """
        _t_start = _time.time()
        _log_diag(f'[V3 Understanding] {step["id"]} | {len(cropped_bboxes)} crops'
                  + (f' + {len(overflow_bboxes)} text-only' if overflow_bboxes else ''))

        resolved_description = self._resolve_placeholders(
            step.get('description', ''), evidence)

        # Build content list
        understand_content = [
            {'type': 'text', 'value': (
                "Examine the image crops and answer the question below.\n\n"
                f"=== CONTEXT FROM PREVIOUS STEPS ===\n{dep_context}\n\n"
                f"=== QUESTION ===\n{resolved_description}\n\n"
                "=== INSTRUCTIONS ===\n"
                "1. Examine each cropped region (labeled with what it contains).\n"
                "2. Use the FULL ORIGINAL IMAGE (attached last) for global "
                "context.\n"
                "3. Output values WITH POSITION INDICES so they can be aligned "
                "later.\n"
                '   Format: "[1]=value1, [2]=value2, [3]=value3, ..."\n'
                '   For structured data: "[row1_col1]=X, [row1_col2]=Y, '
                '[row2_col1]=Z, ..."\n'
                "4. Be specific with values. Include units if present.\n"
                "5. If you cannot determine a value, write [N]=? for that "
                "position.\n"
                "6. Output ONLY the indexed values. No explanation, no reasoning."
            )}
        ]

        # Attach crops with labels (up to max_crops)
        for i, (bbox, label) in enumerate(zip(cropped_bboxes, cropped_labels)):
            understand_content.append(
                {'type': 'text', 'value': f'--- [Region {i+1}: "{label}"] ---'}
            )
            p_bbox = [
                int((bbox[0] / 1000.0) * orig_w),
                int((bbox[1] / 1000.0) * orig_h),
                int((bbox[2] / 1000.0) * orig_w),
                int((bbox[3] / 1000.0) * orig_h),
            ]
            cropped_b64 = self._crop_and_encode_base64(image_path, p_bbox)
            understand_content.append({'type': 'image_base64', 'value': cropped_b64})

        # Overflow regions: text-only (label + bbox), no image
        if overflow_bboxes:
            overflow_lines = []
            for i, (bbox, label) in enumerate(zip(overflow_bboxes, overflow_labels)):
                overflow_lines.append(
                    f"  [{len(cropped_bboxes) + i + 1}] \"{label}\" "
                    f"— bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                )
            overflow_text = (
                f"--- TEXT-ONLY REGIONS (not cropped due to limit of "
                f"{self.max_crops}) ---\n"
                f"The following {len(overflow_bboxes)} regions were identified "
                f"but are NOT shown as images. Use the ORIGINAL FULL IMAGE below "
                f"to locate them.\n"
                + "\n".join(overflow_lines)
            )
            understand_content.append({'type': 'text', 'value': overflow_text})

        # Original image for global context
        orig_thumb_b64 = self._encode_pil_to_base64(
            Image.open(image_path).convert('RGB'), max_edge=1024)
        understand_content.append(
            {'type': 'text',
             'value': "--- [FULL ORIGINAL IMAGE (for global context)] ---"}
        )
        understand_content.append({'type': 'image_base64', 'value': orig_thumb_b64})

        # Disable image-processing kwargs for understanding (crops are pre-sized)
        saved_extra_body = self.extra_body
        self.extra_body = {}
        try:
            ret, understand_text, resp = super().generate_inner(understand_content)
        finally:
            self.extra_body = saved_extra_body

        _t_elapsed = _time.time() - _t_start
        _log_diag(f'[V3 Understanding] {step["id"]} done in {_t_elapsed:.0f}s | '
                  f'{len(understand_text) if understand_text else 0} chars')

        if ret != 0 or not understand_text:
            return None

        return understand_text.strip()

    def _execute_compute(self, step: Dict, evidence: Dict[str, Dict]) -> Any:
        """Execute a compute step via LLM."""
        _log_diag(f'[V3 Compute] {step["id"]}: {step.get("description", "")}')

        expression = step.get('expression', step.get('description', ''))

        compute_prompt = (
            "Compute the following using the provided values.\n\n"
            "=== KNOWN VALUES ===\n"
        )
        for dep_id in step['depends_on']:
            ev = evidence.get(dep_id, {})
            val = ev.get('raw_answer') or ev.get('value')
            if val:
                compute_prompt += f"Step {dep_id}: {val}\n"

        compute_prompt += (
            f"\n=== COMPUTATION ===\n"
            f"Task: {step['description']}\n"
            f"Expression: {expression}\n\n"
            "Output ONLY the numerical result. No explanation."
        )

        ret, comp_text, resp = super().generate_inner(
            [{'type': 'text', 'value': compute_prompt}]
        )
        if ret == 0 and comp_text:
            result_val, _ = clean_value(
                extract_final_value(comp_text) or comp_text)
            _log_diag(f'[V3 Compute] {step["id"]} DONE (LLM): {result_val}')
            return result_val if result_val is not None else comp_text.strip()
        return None

    def _execute_logical(self, step: Dict, evidence: Dict[str, Dict]) -> Any:
        """Execute a logical step (filter/sort/group) via LLM."""
        _log_diag(f'[V3 Logical] {step["id"]}: {step.get("description", "")}')

        context = "=== AVAILABLE DATA ===\n"
        for dep_id in step['depends_on']:
            ev = evidence.get(dep_id, {})
            val = ev.get('raw_answer') or ev.get('value')
            if val:
                context += f"Step {dep_id}: {val}\n"

        logical_prompt = (
            f"{context}\n\n"
            f"=== TASK ===\n{step['description']}\n\n"
            "Answer the question concisely. Output ONLY the answer."
        )

        ret, log_text, resp = super().generate_inner(
            [{'type': 'text', 'value': logical_prompt}]
        )
        if ret == 0 and log_text:
            _log_diag(f'[V3 Logical] {step["id"]} DONE (LLM): '
                      f'{log_text[:120]}')
            return log_text.strip()
        return None

    def sequential_execute(self, plan_steps: List[Dict], image_path: str,
                           question: str, orig_w: int, orig_h: int) -> Dict[str, Dict]:
        """Phase 2: Sequential execution loop with DAG topological ordering.

        NO re-planning — failed steps stay FAILED.
        """
        _t_start = _time.time()
        _log_diag('[V3 Phase2] Sequential Execution start')

        states = {s['id']: 'PENDING' for s in plan_steps}
        evidence: Dict[str, Dict] = {}
        all_steps = list(plan_steps)

        iteration = 0
        total_grounding_calls = 0
        total_understanding_calls = 0
        total_compute_llm_calls = 0
        total_logical_llm_calls = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Select ONE ready step
            ready = []
            for s in all_steps:
                if states[s['id']] != 'PENDING':
                    continue
                if all(states.get(dep) == 'DONE' for dep in s['depends_on']):
                    ready.append(s)

            if not ready:
                pending = [s for s in all_steps if states[s['id']] == 'PENDING']
                if not pending:
                    _log_diag('[V3 Loop] All steps completed')
                    break
                # Stuck — try unblocking steps with no deps
                ready = [s for s in pending if not s['depends_on']]
                if not ready:
                    _log_diag(f'[V3 Loop] WARNING: {len(pending)} steps stuck '
                              f'with unmet dependencies')
                    break

            ready.sort(key=lambda s: (s.get('priority', 99), s['id']))
            step = ready[0]
            sid = step['id']
            _log_diag(f'[V3 Loop] Iter {iteration}/{self.max_iterations} | '
                      f'Step {sid} ({step["type"]})')

            states[sid] = 'EXECUTING'

            # Build dependency context (pure text)
            dep_context_lines = []
            for dep_id in step['depends_on']:
                ev = evidence.get(dep_id, {})
                if ev.get('raw_answer'):
                    dep_context_lines.append(
                        f"[Result of Step {dep_id}]: {ev['raw_answer']}")
                elif ev.get('value') is not None:
                    dep_context_lines.append(
                        f"[Result of Step {dep_id}]: {ev['value']}")
            dep_context = ("\n".join(dep_context_lines) if dep_context_lines
                          else "(No previous results)")

            # ============================================================
            # Execute: <ground-understand>
            # ============================================================
            if step['type'] == '<ground-understand>':
                # --- Grounding ---
                total_grounding_calls += 1
                valid_bboxes, valid_labels = self._ground_step(
                    step, image_path, orig_w, orig_h, dep_context, evidence)

                if not valid_bboxes:
                    _log_diag(f'[V3 Loop] {sid} FAILED: no valid coordinates '
                              f'from grounding')
                    evidence[sid] = {
                        'label': step.get('description', ''),
                        'value': None,
                        'raw_answer': 'GROUNDING_FAILED: no valid coordinates',
                        'bboxes': [],
                        'sub_fields': {},
                    }
                    states[sid] = 'FAILED'
                    continue

                # Split into crops and overflow
                if self.max_crops is not None and len(valid_bboxes) > self.max_crops:
                    cropped_bboxes = valid_bboxes[:self.max_crops]
                    cropped_labels = valid_labels[:self.max_crops]
                    overflow_bboxes = valid_bboxes[self.max_crops:]
                    overflow_labels = valid_labels[self.max_crops:]
                else:
                    cropped_bboxes = valid_bboxes
                    cropped_labels = valid_labels
                    overflow_bboxes = []
                    overflow_labels = []

                if overflow_bboxes:
                    _log_diag(f'[V3 Loop] {sid}: {len(valid_bboxes)} bboxes — '
                              f'{len(cropped_bboxes)} crops, '
                              f'{len(overflow_bboxes)} text-only')

                # --- Understanding ---
                total_understanding_calls += 1
                understand_text = self._understand_step(
                    step, image_path, orig_w, orig_h,
                    cropped_bboxes, cropped_labels,
                    overflow_bboxes, overflow_labels,
                    dep_context, evidence)

                if understand_text is None:
                    _log_diag(f'[V3 Loop] {sid} FAILED: understanding returned '
                              f'empty response')
                    evidence[sid] = {
                        'label': step.get('description', ''),
                        'value': None,
                        'raw_answer': 'UNDERSTANDING_FAILED: empty response',
                        'bboxes': cropped_bboxes,
                        'bbox_labels': cropped_labels,
                        'sub_fields': {},
                    }
                    states[sid] = 'FAILED'
                    continue

                evidence[sid] = {
                    'label': step.get('description', ''),
                    'value': understand_text,
                    'raw_answer': understand_text,
                    'bboxes': cropped_bboxes,
                    'bbox_labels': cropped_labels,
                    'sub_fields': {},
                }
                states[sid] = 'DONE'
                _log_diag(f'[V3 Loop] {sid} DONE: {understand_text[:120]}...')

            # ============================================================
            # Execute: <compute>
            # ============================================================
            elif step['type'] == '<compute>':
                total_compute_llm_calls += 1
                result = self._execute_compute(step, evidence)
                if result is not None:
                    evidence[sid] = {
                        'label': step['description'],
                        'value': result,
                        'raw_answer': str(result),
                        'bboxes': [],
                        'sub_fields': {},
                    }
                    states[sid] = 'DONE'
                else:
                    states[sid] = 'FAILED'
                    _log_diag(f'[V3 Loop] {sid} FAILED: compute')

            # ============================================================
            # Execute: <logical>
            # ============================================================
            elif step['type'] == '<logical>':
                total_logical_llm_calls += 1
                result = self._execute_logical(step, evidence)
                if result is not None:
                    evidence[sid] = {
                        'label': step['description'],
                        'value': result,
                        'raw_answer': str(result),
                        'bboxes': [],
                        'sub_fields': {},
                    }
                    states[sid] = 'DONE'
                else:
                    states[sid] = 'FAILED'
                    _log_diag(f'[V3 Loop] {sid} FAILED: logical')

            else:
                _log_diag(f'[V3 Loop] {sid} FAILED: unknown type '
                          f'"{step["type"]}"')
                states[sid] = 'FAILED'

            # Check if done
            pending_after = [s for s in all_steps
                           if states[s['id']] == 'PENDING']
            if not pending_after:
                break

        _t_elapsed = _time.time() - _t_start
        done_count = sum(1 for v in states.values() if v == 'DONE')
        failed_count = sum(1 for v in states.values() if v == 'FAILED')
        pending_count = sum(1 for v in states.values() if v == 'PENDING')
        _log_diag(f'[V3 Phase2] Done in {_t_elapsed:.0f}s: '
                  f'{done_count} done, {failed_count} failed, '
                  f'{pending_count} pending | '
                  f'{total_grounding_calls} grounding, '
                  f'{total_understanding_calls} understanding, '
                  f'{total_compute_llm_calls} compute-LLM, '
                  f'{total_logical_llm_calls} logical-LLM')

        return evidence

    # ============================================================
    # Phase 3: Final Reasoning
    # ============================================================

    def final_reasoning(self, evidence: Dict[str, Dict], image_path: str,
                        question: str, orig_w: int, orig_h: int) -> str:
        """Phase 3: Generate the final answer using collected evidence + original image."""
        _t_start = _time.time()
        _log_diag('[V3 Phase3] Final Reasoning start')

        # Build evidence summary
        ev_summary_lines = []
        for sid in sorted(evidence.keys()):
            ev = evidence[sid]
            raw = ev.get('raw_answer') or str(ev.get('value', ''))
            if raw and raw != 'None':
                if len(raw) > 500:
                    raw = raw[:500] + "..."
                ev_summary_lines.append(
                    f"[Step {sid} — {ev.get('label', sid)}]: {raw}")

        evidence_summary = ("\n".join(ev_summary_lines) if ev_summary_lines
                           else "(No evidence collected)")

        final_prompt = (
            "You are solving a visual reasoning problem. Below is the evidence "
            "collected from step-by-step analysis and the original image.\n\n"
            "=== COLLECTED EVIDENCE ===\n"
            f"{evidence_summary}\n\n"
            "=== ORIGINAL QUESTION ===\n"
            f"<q>{question}</q>\n\n"
            "=== INSTRUCTIONS ===\n"
            "1. The evidence contains data extracted from the image. TRUST "
            "these values.\n"
            "2. You MAY use the original image below to verify or supplement "
            "the evidence.\n"
            "3. Show your calculation steps concisely. Under 300 words.\n"
            "4. At the VERY END, enclose ONLY the final answer in "
            "<a>...</a> tags.\n"
            "5. Output the answer in the format requested by the original "
            "question."
        )

        final_content = [{'type': 'text', 'value': final_prompt}]

        # Attach original image (thumbnail)
        orig_thumb_b64 = self._encode_pil_to_base64(
            Image.open(image_path).convert('RGB'), max_edge=1024)
        final_content.append({'type': 'image_base64', 'value': orig_thumb_b64})

        # Disable image-processing kwargs for final reasoning
        saved_extra_body = self.extra_body
        self.extra_body = {}
        try:
            ret, answer, resp = super().generate_inner(final_content)
        finally:
            self.extra_body = saved_extra_body

        _t_elapsed = _time.time() - _t_start
        _log_diag(f'[V3 Phase3] Done in {_t_elapsed:.0f}s | '
                  f'{len(answer) if answer else 0} chars')

        return answer if ret == 0 else ""

    # ============================================================
    # Main Entry Point: generate_inner
    # ============================================================

    def generate_inner(self, inputs, **kwargs):
        """Main entry point called by the VLMEvalKit framework.

        Orchestrates three phases:
        Phase 1: Planning → Phase 2: Execute → Phase 3: Reason.
        If planning fails, falls back to direct image+question reasoning.
        """
        _t_total_start = _time.time()

        # 1. Extract image path and question from inputs
        image_path = None
        original_question = ""
        for item in inputs:
            if item['type'] == 'image':
                image_path = item['value']
            elif item['type'] == 'text':
                original_question += item['value']

        if not image_path:
            _log_diag('[V3] No image in inputs, falling back to parent')
            return super().generate_inner(inputs, **kwargs)

        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        # ---- Phase 1: Structured Planning ----
        plan_steps, plan_text = self.structured_planning(
            image_path, original_question)

        # If planning failed (no steps), FALLBACK to direct reasoning
        if not plan_steps:
            _log_diag('[V3 PLANNING_FAIL] No valid plan JSON — falling back to '
                      'direct reasoning')
            FALLBACK = True
            plan_steps = []
            evidence = {}
        else:
            _log_diag(f'[V3 PLANNING_OK] {len(plan_steps)} steps')
            FALLBACK = False

        if FALLBACK:
            # ---- Fallback: direct image + question reasoning ----
            _t_start = _time.time()
            _log_diag('[V3 Phase3] Fallback Reasoning start')

            final_prompt = (
                "Answer the following question based on the image.\n\n"
                "=== QUESTION ===\n"
                f"<q>{original_question}</q>\n\n"
                "=== INSTRUCTIONS ===\n"
                "1. Look at the image carefully and reason step by step.\n"
                "2. At the VERY END, enclose ONLY the final answer in "
                "<a>...</a> tags.\n"
                "3. Output the answer in the format requested by the question."
            )

            final_content = [{'type': 'text', 'value': final_prompt}]
            orig_thumb_b64 = self._encode_pil_to_base64(
                Image.open(image_path).convert('RGB'), max_edge=1024)
            final_content.append({'type': 'image_base64', 'value': orig_thumb_b64})

            saved_extra_body = self.extra_body
            self.extra_body = {}
            try:
                ret, answer, resp = super().generate_inner(final_content)
            finally:
                self.extra_body = saved_extra_body

            _t_elapsed = _time.time() - _t_start
            _log_diag(f'[V3 Phase3] Fallback done in {_t_elapsed:.0f}s')
            _t_total = _time.time() - _t_total_start
            _log_diag(f'[V3] Total (fallback): {_t_total:.0f}s')
            return ret if ret != 0 else 0, answer, resp

        # ---- Phase 2: Sequential Execution Loop ----
        evidence = self.sequential_execute(
            plan_steps, image_path, original_question, orig_w, orig_h
        )

        # ---- Phase 3: Final Reasoning ----
        answer = self.final_reasoning(
            evidence, image_path, original_question, orig_w, orig_h
        )

        _t_total = _time.time() - _t_total_start
        done_count = sum(1 for ev in evidence.values()
                        if ev.get('value') is not None)
        _log_diag(f'[V3] Total: {_t_total:.0f}s | {len(plan_steps)} plan '
                  f'steps | {done_count} evidence items')

        return 0, answer, None

    # ============================================================
    # Answer Extraction (used by framework evaluation)
    # ============================================================

    @staticmethod
    def extract_final_answer(text: str) -> str:
        """Extract the final answer from model output using <a>...</a> tags.

        Falls back to the last non-empty line.
        """
        if not text:
            return ""

        # Try <a>...</a> tags first
        match = re.search(r'<a>(.*?)</a>', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: last non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[-1] if lines else ""
