"""
OCR-Reasoning Agent v2 — Iterative Execute-Verify Loop

Architecture:
  Phase 1: Structured Planning (JSON schema with DAG dependencies)
  Phase 2: Iterative Execute-Verify Loop
    - DAG scheduling: select ready steps each iteration
    - Batch Grounding: one API call for all pending reads
    - Grounding Validation: bbox dedup, area filtering, anomaly detection
    - Batch Reading: one API call to extract values from all crops
    - Compute Execution: safe eval or LLM-based computation
    - Sufficiency Check: evidence → answerable? if not → Re-plan
  Phase 3: Final Reasoning with complete evidence chain

Key improvements over v1 (ThreeStageOCRAgent):
  1. Iterative loop: can dynamically add steps when evidence is insufficient
  2. Grounding quality validation: bbox dedup, area filtering, anomaly detection
  3. Separation of grounding and reading: ground first, then OCR on crops
  4. Evidence store with layered context management
  5. Chart-aware planning prompts
  6. DAG-based step dependency resolution
"""

import os
import io
import re
import ast
import json
import base64
import operator
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

from PIL import Image

from vlmeval.api.gpt import OpenAIWrapper, _log_diag


# ============================================================
# Data Structures
# ============================================================

class StepState(Enum):
    PENDING = "pending"
    READY = "ready"
    EXECUTING = "executing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class PlanStep:
    """A single atomic step in the execution plan."""
    id: str                              # "s1", "s2", ...
    type: str                            # "ground_and_read" | "compute" | "verify"
    description: str                     # human-readable description
    target_desc: str = ""                # (read) semantic location description for grounding
    expected_type: str = "text"          # "number" | "text" | "option" | "integer" | "float"
    expected_unit: str = ""              # e.g. "元", "毫米", "%"
    expression: str = ""                 # (compute) expression or operation description
    depends_on: List[str] = field(default_factory=list)
    priority: int = 1


@dataclass
class Evidence:
    """Collected evidence from a completed step."""
    step_id: str
    label: str                           # semantic label (what this evidence represents)
    value: Any = None                    # extracted/computed value
    confidence: float = 1.0              # 0.0 - 1.0
    bbox_norm: Optional[List[int]] = None  # normalized [xmin, ymin, xmax, ymax] in [0,1000]
    raw_text: str = ""                   # raw model output for this step
    unit: str = ""                       # unit of the value

    def compact_str(self) -> str:
        """Compact representation for context compression."""
        bbox_str = ""
        if self.bbox_norm:
            bbox_str = f" @ [{self.bbox_norm[0]},{self.bbox_norm[1]},{self.bbox_norm[2]},{self.bbox_norm[3]}]"
        val_str = f"{self.value}"
        if self.unit:
            val_str += f" {self.unit}"
        conf_str = ""
        if self.confidence < 0.8:
            conf_str = " [LOW CONFIDENCE]"
        return f"[{self.label}] = {val_str}{bbox_str}{conf_str}"


class EvidenceStore:
    """Layered evidence management with context compression."""

    def __init__(self):
        self._items: List[Evidence] = []
        self._by_step: Dict[str, Evidence] = {}

    def add(self, evidence: Evidence):
        self._items.append(evidence)
        self._by_step[evidence.step_id] = evidence

    def get(self, step_id: str) -> Optional[Evidence]:
        return self._by_step.get(step_id)

    def get_all(self) -> List[Evidence]:
        return list(self._items)

    def get_read_evidence(self) -> List[Evidence]:
        """Return only evidence from ground_and_read steps (visual extraction)."""
        return [e for e in self._items if e.bbox_norm is not None]

    def compress_for_context(self, max_items: int = 20) -> str:
        """Compress evidence into a compact text block for LLM context.

        Recent items are kept in full; older items are compressed to one-liners.
        """
        if not self._items:
            return "(No evidence collected yet)"

        lines = []
        n = len(self._items)
        for i, ev in enumerate(self._items):
            is_recent = (n - i) <= max_items
            if is_recent:
                lines.append(ev.compact_str())
            else:
                # Ultra-compact: just label + value
                short = f"[{ev.label}] = {ev.value}"
                if ev.unit:
                    short += f" {ev.unit}"
                lines.append(short)
        return "\n".join(lines)

    def to_value_map(self) -> Dict[str, Any]:
        """Return a dict of {step_id: value} for computation."""
        return {ev.step_id: ev.value for ev in self._items if ev.value is not None}

    def has_all(self, step_ids: List[str]) -> bool:
        """Check if all required step_ids have evidence."""
        return all(sid in self._by_step for sid in step_ids)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"EvidenceStore({len(self._items)} items)"


# ============================================================
# Safe Compute Sandbox
# ============================================================

_SAFE_OPS = {
    'add': operator.add, 'sub': operator.sub, 'mul': operator.mul,
    'truediv': operator.truediv, 'floordiv': operator.floordiv,
    'pow': operator.pow, 'mod': operator.mod,
    'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum,
    'int': int, 'float': float, 'len': len,
}


def clean_value(value_str: str):
    """Strip currency symbols, whitespace, and noise from a value string.
    Returns (cleaned_value, was_cleaned_flag).
    """
    if not value_str:
        return None, False
    original = value_str
    # Strip common currency/unit symbols
    for sym in ['$', '€', '¥', '£', '￥', '₹', '%', '°']:
        value_str = value_str.replace(sym, '')
    # Strip parenthetical notes like "(conf=1.0)" or "(low confidence)"
    value_str = re.sub(r'\s*\(.*?\)\s*$', '', value_str)
    # Strip trailing punctuation and whitespace
    value_str = value_str.strip().rstrip('.,;:')
    was_cleaned = (value_str != original.strip())
    if not value_str:
        return None, True
    # Try numeric conversion
    try:
        if '.' in value_str:
            return float(value_str), was_cleaned
        else:
            return int(value_str), was_cleaned
    except ValueError:
        return value_str, was_cleaned


def extract_final_value(text: str) -> Optional[str]:
    """Extract the actual value from verbose model output (handles thinking traces).

    Qwen thinking models often output: response... value (conf=0.8)
    This function strips reasoning and returns only the final value.
    """
    if not text:
        return None

    # Step 1: Extract content after last thinking/response tag
    for tag in ['<think>', '<｜end▁of▁thinking｜>', ' response']:
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

    # Step 2: Remove Qwen prefix tags
    text = re.sub(r'^/no_think\s*', '', text)
    text = re.sub(r'^/think\s*', '', text)

    # Step 3: Take the last non-empty, short-ish line as the value
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None

    short_lines = [l for l in lines if len(l) < 200]
    if short_lines:
        text = short_lines[-1]
    else:
        text = lines[-1]

    # Step 4: Remove trailing markers like "(conf=0.8)"
    text = re.sub(r'\s*\(conf[^)]*\)\s*$', '', text)
    text = text.strip()

    return text


def safe_eval(expression: str, variables: Dict[str, Any]) -> Any:
    """Safely evaluate a mathematical expression with given variables.

    Uses Python's eval with restricted builtins for basic arithmetic.
    Falls back to returning the expression string if evaluation is unsafe.
    """
    # Whitelist of allowed names
    allowed_names = set(_SAFE_OPS.keys()) | set(variables.keys())

    # Clean the expression: replace $s1 with s1 style references
    expr = expression
    expr = re.sub(r'\$(\w+)', r'\1', expr)

    try:
        # Check all names in the AST are allowed
        tree = ast.parse(expr, mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                raise NameError(f"'{node.id}' is not allowed")

        # Build safe globals
        safe_globals = {"__builtins__": {}}
        safe_globals.update(_SAFE_OPS)

        result = eval(compile(tree, '<string>', 'eval'), safe_globals, variables)
        return result
    except Exception:
        # If safe eval fails, return a marker that LLM execution is needed
        return None


# ============================================================
# Main Agent Class
# ============================================================

class OCRReasoningAgentV2(OpenAIWrapper):
    """OCR-Reasoning Agent with iterative execute-verify loop.

    Extends OpenAIWrapper for full compatibility with the VLMEvalKit framework.
    Overrides generate_inner() to implement the 3-phase agent pipeline.
    """

    def __init__(self,
                 padding: int = 60,
                 scale_factor: float = 3.0,
                 inspect_padding: int = 150,
                 inspect_scale: float = 1.5,
                 max_crops: int = 12,
                 max_crop_edge: int = 1024,
                 max_iterations: int = 20,
                 planner_model: str = None,
                 planner_api_base: str = None,
                 grounder_model: str = None,
                 grounder_api_base: str = None,
                 **kwargs):
        """
        Args:
            padding: pixel padding for tight crops (ground_and_read)
            scale_factor: magnification for tight crops (ground_and_read)
            inspect_padding: pixel padding for generous crops (inspect_region)
            inspect_scale: magnification for generous crops (inspect_region)
            max_crops: maximum number of cropped sub-images per iteration
            max_crop_edge: max edge length (px) for crop images
            max_iterations: max iterations for the execute-verify loop
            planner_model: override model name for planning (defaults to parent's model)
            planner_api_base: override API base for planning
            grounder_model: override model name for grounding
            grounder_api_base: override API base for grounding
        """
        super().__init__(**kwargs)
        self.padding = padding
        self.scale_factor = scale_factor
        self.inspect_padding = inspect_padding
        self.inspect_scale = inspect_scale
        self.max_crops = max_crops
        self.max_crop_edge = max_crop_edge
        self.max_iterations = max_iterations

        if 'max_tokens' not in kwargs:
            self.max_tokens = 6144

        # ---- Planner instance (Phase 1 & re-planning) ----
        planner_kwargs = dict(
            model=planner_model or self.model,
            api_base=planner_api_base or self.api_base,
            key=getattr(self, 'key', 'sk-EMPTY'),
            temperature=0.0,
            max_tokens=6144,
            img_size=self.img_size,
            img_detail=self.img_detail,
            extra_body=self.extra_body,
            verbose=self.verbose,
        )
        self.planner = OpenAIWrapper(**planner_kwargs)

        # ---- Grounder instance (Phase 2: batch grounding) ----
        grounder_kwargs = dict(
            model=grounder_model or self.model,
            api_base=grounder_api_base or self.api_base,
            key=getattr(self, 'key', 'sk-EMPTY'),
            temperature=0.0,
            max_tokens=6144,
            img_size=self.img_size,
            img_detail=self.img_detail,
            extra_body=self.extra_body,
            verbose=self.verbose,
        )
        self.grounder = OpenAIWrapper(**grounder_kwargs)

    # ============================================================
    # Image Utilities (in-memory crop + encode, same as v1)
    # ============================================================

    def _crop_and_encode_base64(self, image_path: str, physical_bbox: List[int],
                                 max_edge: Optional[int] = None) -> str:
        """Crop image at physical coordinates, magnify, limit max edge, return base64."""
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
                new_size = (int(cropped.width * self.scale_factor),
                            int(cropped.height * self.scale_factor))
                cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)

            if max_edge is not None:
                w, h = cropped.size
                _max = max(w, h)
                if _max > max_edge:
                    ratio = max_edge / _max
                    cropped = cropped.resize((int(w * ratio), int(h * ratio)),
                                             Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            cropped.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

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

    def _build_image_content(self, b64_string: str) -> Dict:
        """Build an OpenAI API image_url content block from base64."""
        return {
            'type': 'image_url',
            'image_url': {
                'url': f"data:image/jpeg;base64,{b64_string}"
            }
        }

    def _build_text_content(self, text: str) -> Dict:
        """Build an OpenAI API text content block."""
        return {'type': 'text', 'text': text}

    # ============================================================
    # JSON Parsing (Robust)
    # ============================================================

    @staticmethod
    def _parse_json_robust(text: str) -> Optional[Dict]:
        """Robustly extract JSON from model output (handles thinking traces)."""
        if not text:
            return None

        # Step 0: Strip thinking/response tags and their content
        for tag in ['<think>', ' response', ' response']:
            for end_tag in ['</think>', '</think>', '</think>']:
                if end_tag in text:
                    parts = text.rsplit(end_tag, 1)
                    text = parts[-1]
                if tag in text and end_tag in text:
                    text = re.sub(re.escape(tag) + r'.*?' + re.escape(end_tag), '', text, flags=re.DOTALL)

        # Step 1: Strip markdown code fences and prefixes
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = re.sub(r'^/no_think\s*', '', text)
        text = re.sub(r'^/think\s*', '', text)

        # Step 2: Try to find and extract JSON
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
                json_lines = [l for l in lines if re.search(r'[{}\[\]:"]', l) or l.strip().startswith(('{','}','[',']'))]
                text = '\n'.join(json_lines)
            elif attempt == 3:
                text = re.sub(r'(\s+)(\w+)(\s*):', r'\1"\2"\3:', text)
        return None

    # ============================================================
    # Phase 1: Structured Planning
    # ============================================================

    def _build_planning_prompt(self, question: str) -> str:
        """Build the planning prompt that instructs the model to output a JSON plan."""
        return (
            "/no_think\n"
            f"Create a JSON execution plan to answer this question: <q>{question}</q>\n\n"
            "=== YOUR ROLE ===\n"
            "You are a planner. Output a step-by-step plan as JSON. Do NOT reason or explain — output ONLY the JSON.\n\n"
            "=== STEP TYPES ===\n"
            '- "ground_and_read": Locate a specific text/number at a precise position and read its exact value.\n'
            '  Best for: prices, names, labels, table cells, form fields, individual numbers, legend items.\n'
            '  The reader will see a tight, magnified crop of just the target text.\n'
            '- "inspect_region": Visually analyze a region to count, check, or compare visual features.\n'
            '  Best for: counting colored segments in a bar, checking if a color/pattern exists,\n'
            '  reading a trend from a line, comparing visual attributes.\n'
            '  The reader will see a generous crop showing full context around the target.\n'
            '- "compute": Calculate using previously collected values.\n\n'
            "=== CRITICAL RULES ===\n"
            "1. NO PRE-FILTERING. List EVERY single item/candidate in the image. Do NOT skip any.\n"
            "   The reasoning phase will decide which items are relevant — NOT you.\n"
            "2. Choose the correct step type:\n"
            "   - Reading a specific printed number/text at a known location → ground_and_read\n"
            "   - Counting, checking presence/absence, visual analysis of a region → inspect_region\n"
            "3. ONE visual element per step. Be specific about WHAT and WHERE.\n"
            "4. Raw data steps first, then compute steps.\n"
            "5. Max 25 steps.\n\n"
            "=== JSON SCHEMA ===\n"
            '{\n'
            '  "steps": [\n'
            '    {"id": "s1", "type": "ground_and_read", "description": "...", '
            '"target_desc": "precise location of text/number", "expected_type": "number|text", '
            '"expected_unit": "...", "depends_on": [], "priority": 1},\n'
            '    {"id": "s2", "type": "inspect_region", "description": "Count the number of...", '
            '"target_desc": "the region/bar for...", "expected_type": "number|text", '
            '"expected_unit": "...", "depends_on": [], "priority": 1},\n'
            '    {"id": "sN", "type": "compute", "description": "...", '
            '"expression": "s1 + s2", "depends_on": ["s1", "s2"], "priority": 2}\n'
            '  ]\n'
            '}\n'
            "OUTPUT ONLY THE JSON. No thinking, no explanation, no markdown — just the JSON object."
        )

    def _parse_plan(self, plan_json: Dict) -> List[PlanStep]:
        """Parse a planning JSON dict into a list of PlanStep objects."""
        steps = []
        raw_steps = plan_json.get('steps', [])
        for i, raw in enumerate(raw_steps):
            step = PlanStep(
                id=raw.get('id', f's{i+1}'),
                type=raw.get('type', 'ground_and_read'),
                description=raw.get('description', ''),
                target_desc=raw.get('target_desc', ''),
                expected_type=raw.get('expected_type', 'text'),
                expected_unit=raw.get('expected_unit', ''),
                expression=raw.get('expression', ''),
                depends_on=raw.get('depends_on', []),
                priority=raw.get('priority', i + 1),
            )
            steps.append(step)
        return steps

    def structured_planning(self, image_path: str, question: str) -> Tuple[List[PlanStep], str]:
        """Phase 1: Generate a structured execution plan from the image + question.

        Returns:
            (plan_steps, raw_plan_text) — parsed steps and raw model output.
        """
        _t_start = _time.time()
        _log_diag('[V2 Phase1] Structured Planning start')

        prompt = self._build_planning_prompt(question)
        plan_inputs = [
            {'type': 'text', 'value': prompt},
            {'type': 'image', 'value': image_path},
        ]

        ret, plan_text, resp = self.planner.generate_inner(plan_inputs)
        _t_elapsed = _time.time() - _t_start

        if ret != 0:
            _log_diag(f'[V2 Phase1] FAILED after {_t_elapsed:.0f}s')
            return [], plan_text

        plan_json = self._parse_json_robust(plan_text)
        if plan_json is None:
            _log_diag(f'[V2 Phase1] JSON parse failed. Raw: {plan_text[:300]}...')
            return [], plan_text

        steps = self._parse_plan(plan_json)
        chart_type = plan_json.get('chart_type', None)
        chart_info = f' chart_type={chart_type}' if chart_type else ''
        _log_diag(f'[V2 Phase1] Done in {_t_elapsed:.0f}s | {len(steps)} steps{chart_info}')

        return steps, plan_text

    # ============================================================
    # Phase 2: Iterative Execute-Verify Loop
    # ============================================================

    def _get_ready_steps(self, steps: List[PlanStep],
                         states: Dict[str, StepState],
                         evidence: EvidenceStore) -> List[PlanStep]:
        """Select steps whose dependencies are all DONE.

        DAG topological selection: a step is READY if all its depends_on
        steps are in DONE state.
        """
        ready = []
        for step in steps:
            if states.get(step.id) != StepState.PENDING:
                continue
            if all(states.get(dep) == StepState.DONE for dep in step.depends_on):
                ready.append(step)
        # Sort by priority then by id
        ready.sort(key=lambda s: (s.priority, s.id))
        return ready

    def _ground_single(self, step: PlanStep, image_path: str,
                      orig_w: int, orig_h: int,
                      plan_text: str, question: str) -> Optional[Dict]:
        """Ground a SINGLE target element. Returns {label, bbox_norm, confidence, raw_line} or None."""
        _t_start = _time.time()
        _log_diag(f'[V2 Grounding] Grounding single target: {step.id}')

        grounding_prompt = (
            f"Locate this visual element in the image.\n\n"
            f"TARGET: {step.target_desc}\n\n"
            "Copy this line EXACTLY, replacing each ? with a coordinate integer in [0, 1000]:\n"
            f"\"{step.id}: {step.target_desc}\" -> [?, ?, ?, ?]\n\n"
            "RULES:\n"
            "1. Replace ? with integers in [0, 1000] (normalized to image dimensions).\n"
            "2. Box the TIGHTEST rectangle around the target text/number.\n"
            '3. If NOT visible, replace all four ? with -1.\n'
            "4. The LAST line of your response must be the completed coordinate line. Nothing else after.\n"
            "5. The line must be character-for-character IDENTICAL except ? \u2192 numbers."
        )

        planning_prompt_compact = (
            f"You are solving: <q>{question}</q>\n"
            "Create an atomic execution plan."
        )
        history = [
            {'role': 'user', 'content': [
                {'type': 'text', 'value': planning_prompt_compact},
                {'type': 'image', 'value': image_path},
            ]},
            {'role': 'assistant', 'content': [{'type': 'text', 'value': plan_text[:2000]}]},
            {'role': 'user', 'content': [{'type': 'text', 'value': grounding_prompt}]},
        ]

        ret, grounding_text, resp = self.grounder.generate_inner(history)
        _t_elapsed = _time.time() - _t_start
        _log_diag(f'[V2 Grounding] Done in {_t_elapsed:.0f}s | {len(grounding_text)} chars')

        if ret != 0 or not grounding_text:
            return None

        # Parse labeled format first, then bare coordinates
        labeled_pattern = r'"([^"]*)"\s*->\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
        matches = re.findall(labeled_pattern, grounding_text)

        if not matches:
            # Bare coordinates: find all [x, y, x, y] patterns
            bare_pattern = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
            bare_matches = re.findall(bare_pattern, grounding_text)
            if bare_matches:
                bm = bare_matches[0]
                label = step.target_desc[:60]
                matches = [(f"{step.id}: {label}",) + bm]

        if not matches:
            _log_diag(f'[V2 Grounding] No coordinates found for {step.id}')
            return None

        m = matches[0]
        raw_label = m[0].strip()
        bbox = [int(m[1]), int(m[2]), int(m[3]), int(m[4])]

        if ':' in raw_label:
            parts = raw_label.split(':', 1)
            parsed_step_id = parts[0].strip()
            label = parts[1].strip()
        else:
            parsed_step_id = raw_label
            label = raw_label

        return {
            'step_id': parsed_step_id,
            'label': label,
            'bbox_norm': bbox,
            'confidence': 1.0,
            'raw_line': f'"{raw_label}" -> {bbox}',
        }


    def _validate_grounding(self, bbox_norm: List[int], orig_w: int, orig_h: int) -> bool:
        """Validate a single grounding result.

        Checks:
          1. Not a null/missing marker
          2. Coordinates within [0, 1000] range
          3. Bbox has positive area
          4. Aspect ratio within reasonable bounds (≤50:1)
        """
        xmin, ymin, xmax, ymax = bbox_norm

        # Null marker check
        if xmin < 0 or ymin < 0:
            return False
        if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
            return False

        # Range check
        if not (0 <= xmin <= 1000 and 0 <= ymin <= 1000 and
                0 <= xmax <= 1000 and 0 <= ymax <= 1000):
            return False

        # Bbox must have positive area
        if xmin >= xmax or ymin >= ymax:
            return False

        # Aspect ratio sanity check (width/height extremes)
        w_norm = (xmax - xmin)
        h_norm = (ymax - ymin)
        if h_norm > 0 and w_norm > 0:
            aspect = max(w_norm / h_norm, h_norm / w_norm)
            if aspect > 50:  # extremely thin bbox — probably wrong
                _log_diag(f'[V2 Validate] Bbox aspect ratio extreme: {aspect:.1f} | {bbox_norm}')
                return False

        return True

    def _read_single_crop(self, grounded_item: Dict, image_path: str,
                         orig_w: int, orig_h: int) -> Optional[Evidence]:
        """Read value from a SINGLE cropped region. Returns Evidence or None."""
        _t_start = _time.time()
        step_id = grounded_item['step_id']
        label = grounded_item['label']
        bbox_norm = grounded_item['bbox_norm']
        _log_diag(f'[V2 Read] Reading single crop: {step_id}')

        expected_type = grounded_item.get('expected_type', 'text')
        expected_unit = grounded_item.get('expected_unit', '')

        read_content = [
            {'type': 'text', 'value': (
                "/no_think\n"
                f"Read the exact value from this cropped region.\n\n"
                f"Target: {label}\n"
                f"Expected type: {expected_type} {expected_unit}\n\n"
                "CRITICAL RULES:\n"
                "1. Output ONLY the value. NOTHING else. No reasoning, no explanation, no thinking.\n"
                "2. Your ENTIRE response must be a single line. Just the value.\n"
                "3. Examples: '2 $', '45.5 kg', 'A'\n"
                "4. If unclear: output UNCLEAR. If not found: output NOT_FOUND."
            )}
        ]

        # Convert normalized [0,1000] to physical pixels
        p_bbox = [
            int((bbox_norm[0] / 1000.0) * orig_w),
            int((bbox_norm[1] / 1000.0) * orig_h),
            int((bbox_norm[2] / 1000.0) * orig_w),
            int((bbox_norm[3] / 1000.0) * orig_h),
        ]
        # Add labels and crop
        read_content.append({
            'type': 'text',
            'value': f'--- Region: [{step_id}] "{label}" ---'
        })
        b64 = self._crop_and_encode_base64(image_path, p_bbox, max_edge=self.max_crop_edge)
        read_content.append({'type': 'image_base64', 'value': b64})

        read_history = [{'role': 'user', 'content': read_content}]
        ret, read_text, resp = super().generate_inner(read_history)

        _t_elapsed = _time.time() - _t_start
        _log_diag(f'[V2 Read] Done in {_t_elapsed:.0f}s | {len(read_text)} chars')

        if ret != 0 or not read_text:
            return None

        value_str = read_text.strip()

        # Extract final value from potential thinking trace
        extracted = extract_final_value(value_str)
        if extracted:
            value_str = extracted

        # Parse the single value
        cleaned_val, was_cleaned = clean_value(value_str)
        if cleaned_val is None:
            conf = 0.3
        elif was_cleaned:
            conf = 0.8
        else:
            conf = 1.0

        # Extract unit
        unit = ''
        if isinstance(cleaned_val, (int, float)):
            raw_no_num = re.sub(r'[\d.]+', '', value_str.replace('$', '').replace('\u20ac', ''))
            unit = raw_no_num.strip().rstrip('.,;:()')

        evidence = Evidence(
            step_id=step_id,
            label=label,
            value=cleaned_val,
            confidence=conf,
            bbox_norm=bbox_norm,
            raw_text=value_str,
            unit=unit,
        )
        _log_diag(f'[V2 Read] Extracted: {evidence.compact_str()}')
        return evidence


    def _execute_compute(self, step: PlanStep, evidence: EvidenceStore) -> Any:
        """Execute a compute step using collected evidence.

        Tries safe_eval first; falls back to LLM-based computation for complex expressions.
        """
        # Gather dependent values
        variables = {}
        for dep_id in step.depends_on:
            ev = evidence.get(dep_id)
            if ev is None or ev.value is None:
                _log_diag(f'[V2 Compute] Missing dependency {dep_id} for step {step.id}')
                return None
            # Use step_id as variable name
            variables[dep_id] = ev.value
            variables[f'{dep_id}_value'] = ev.value

        # Clean variable values: ensure numeric, not strings with symbols
        clean_vars = {}
        for k, v in variables.items():
            if isinstance(v, str):
                cleaned, _ = clean_value(v)
                clean_vars[k] = cleaned if cleaned is not None else v
            else:
                clean_vars[k] = v

        # Try safe mathematical evaluation
        result = safe_eval(step.expression, clean_vars)
        if result is not None:
            _log_diag(f'[V2 Compute] {step.id}: {step.expression} = {result}')
            return result

        # Fallback: use LLM for complex computation
        _log_diag(f'[V2 Compute] {step.id}: safe_eval failed, using LLM')
        compute_prompt = (
            f"Compute the following using these known values:\n"
            + "\n".join(f"  {k} = {v}" for k, v in variables.items()) +
            f"\n\nOperation: {step.description}\n"
            f"Expression: {step.expression}\n\n"
            "Output ONLY the numerical result, nothing else."
        )
        ret, answer, resp = super().generate_inner([{'type': 'text', 'value': compute_prompt}])
        if ret == 0 and answer:
            try:
                if '.' in answer:
                    return float(answer.strip())
                return int(answer.strip())
            except ValueError:
                return answer.strip()
        return None

    def _check_sufficiency(self, evidence: EvidenceStore, steps: List[PlanStep],
                           question: str) -> Tuple[bool, List[str]]:
        """Check if collected evidence is sufficient to answer the question.

        Returns:
            (is_sufficient, missing_descriptions) — whether answerable and what's missing.
        """
        # If all steps are DONE, evidence is sufficient
        # (otherwise we'd still be executing steps)
        all_read_evidence = evidence.get_read_evidence()

        # Check for low-confidence reads
        low_conf = [e for e in all_read_evidence if e.confidence < 0.5]
        if low_conf:
            missing = [f'"{e.label}" (read with low confidence: {e.raw_text})' for e in low_conf]
            _log_diag(f'[V2 Sufficiency] {len(low_conf)} low-confidence reads found')
            return False, missing

        # Check for missing/NULL values
        null_reads = [e for e in all_read_evidence if e.value is None]
        if null_reads:
            missing = [f'"{e.label}" (value is NULL/UNCLEAR)' for e in null_reads]
            _log_diag(f'[V2 Sufficiency] {len(null_reads)} null values found')
            return False, missing

        # If we have evidence, consider it sufficient
        if len(all_read_evidence) > 0:
            return True, []

        return False, ["No visual evidence collected yet"]

    def _replan(self, evidence: EvidenceStore, missing_info: List[str],
                image_path: str, question: str, orig_w: int, orig_h: int) -> List[PlanStep]:
        """Generate new steps to fill evidence gaps.

        Sends current evidence summary + missing info to the planner,
        requesting additional read steps.
        """
        _t_start = _time.time()
        _log_diag(f'[V2 Replan] Generating new steps for {len(missing_info)} missing items')

        evidence_summary = evidence.compress_for_context(max_items=10)
        missing_str = "\n".join(f"  - {m}" for m in missing_info)

        replan_prompt = (
            f"We are solving: <q>{question}</q>\n\n"
            "=== EVIDENCE COLLECTED SO FAR ===\n"
            f"{evidence_summary}\n\n"
            "=== MISSING / UNCERTAIN INFORMATION ===\n"
            f"{missing_str}\n\n"
            "=== TASK ===\n"
            "Generate additional 'ground_and_read' steps to fill the gaps. "
            "Each step must target ONE visual element. Output as JSON array:\n\n"
            '[\n'
            '  {\n'
            '    "id": "s_new1",\n'
            '    "type": "ground_and_read",\n'
            '    "description": "...",\n'
            '    "target_desc": "precise semantic location of the element",\n'
            '    "expected_type": "number"|"text"|"option",\n'
            '    "expected_unit": "...",\n'
            '    "depends_on": [],\n'
            '    "priority": 1\n'
            '  }\n'
            ']\n\n'
            "CRITICAL: Output raw JSON array only. No markdown, no explanation."
        )

        replan_inputs = [{'type': 'text', 'value': replan_prompt}]
        # Include original image for context if needed
        if len(evidence.get_read_evidence()) <= 2:
            replan_inputs.append({'type': 'image', 'value': image_path})

        ret, replan_text, resp = self.planner.generate_inner(replan_inputs)
        _t_elapsed = _time.time() - _t_start

        if ret != 0:
            return []

        replan_json = self._parse_json_robust(replan_text)
        if replan_json is None:
            return []

        # replan_json may be a list (array) or a dict with "steps"
        if isinstance(replan_json, list):
            raw_steps = replan_json
        elif isinstance(replan_json, dict):
            raw_steps = replan_json.get('steps', [])
        else:
            return []

        new_steps = self._parse_plan({'steps': raw_steps})
        _log_diag(f'[V2 Replan] Generated {len(new_steps)} new steps in {_t_elapsed:.0f}s')
        return new_steps

    def iterative_execute(self, steps: List[PlanStep], image_path: str,
                          question: str, orig_w: int, orig_h: int,
                          plan_text: str) -> EvidenceStore:
        """Phase 2: Sequential execute-verify loop.

        Processes ONE step per iteration. After each step completes,
        re-evaluates which steps are ready (their dependencies satisfied).
        This supports dynamic step sequencing where later grounding steps
        may depend on the RESULTS of earlier steps.
        """
        evidence = EvidenceStore()
        states: Dict[str, StepState] = {s.id: StepState.PENDING for s in steps}
        all_steps = list(steps)  # mutable copy for re-plan additions

        iteration = 0
        total_grounding_calls = 0
        total_read_calls = 0

        while iteration < self.max_iterations:
            iteration += 1

            # 1. Select ONE ready step (highest priority first)
            ready = []
            for s in all_steps:
                if states.get(s.id) != StepState.PENDING:
                    continue
                if all(states.get(dep) == StepState.DONE for dep in s.depends_on):
                    ready.append(s)

            if not ready:
                pending = [s for s in all_steps if states[s.id] == StepState.PENDING]
                if not pending:
                    _log_diag('[V2 Loop] All steps completed')
                    break
                # Stuck — try force-unblocking steps with no deps
                ready = [s for s in pending if not s.depends_on]
                if not ready:
                    _log_diag(f'[V2 Loop] WARNING: {len(pending)} steps stuck')
                    break

            ready.sort(key=lambda s: (s.priority, s.id))
            step = ready[0]  # Process ONE step per iteration
            states[step.id] = StepState.EXECUTING
            _log_diag(f'[V2 Loop] Iter {iteration}/{self.max_iterations} | Step {step.id} ({step.type})')

            # 2. Execute the single step
            if step.type == 'ground_and_read':
                total_grounding_calls += 1
                gr = self._ground_single(step, image_path, orig_w, orig_h, plan_text, question)

                if gr is None:
                    _log_diag(f'[V2 Loop] No grounding result for {step.id}')
                    evidence.add(Evidence(
                        step_id=step.id, label=step.target_desc,
                        value=None, confidence=0.0, bbox_norm=None,
                        raw_text='NO_GROUNDING_RESULT',
                    ))
                    states[step.id] = StepState.FAILED
                    continue

                # Validate single bbox
                if not self._validate_grounding(gr['bbox_norm'], orig_w, orig_h):
                    _log_diag(f'[V2 Loop] Grounding validation FAILED for {step.id}: {gr["bbox_norm"]}')
                    evidence.add(Evidence(
                        step_id=step.id, label=gr['label'],
                        value=None, confidence=0.1, bbox_norm=gr['bbox_norm'],
                        raw_text=f'VALIDATION_FAILED: {gr["raw_line"]}',
                    ))
                    states[step.id] = StepState.FAILED
                    continue

                # Read single crop
                total_read_calls += 1
                grounded_item = {
                    'step_id': step.id,
                    'label': gr['label'],
                    'bbox_norm': gr['bbox_norm'],
                    'confidence': gr.get('confidence', 1.0),
                    'expected_type': step.expected_type,
                    'expected_unit': step.expected_unit,
                }
                ev = self._read_single_crop(grounded_item, image_path, orig_w, orig_h)
                if ev is not None:
                    evidence.add(ev)
                    states[step.id] = StepState.DONE
                    _log_diag(f'[V2 Loop] {step.id} DONE: {ev.compact_str()}')
                else:
                    states[step.id] = StepState.FAILED
                    _log_diag(f'[V2 Loop] {step.id} FAILED: read returned nothing')

            elif step.type == 'inspect_region':
                # Visual analysis — grounding + generous crop + analysis question
                _t_start = _time.time()
                total_grounding_calls += 1
                _log_diag(f'[V2 Inspect] Inspecting region: {step.id}')

                grounding_prompt = (
                    f"Locate this visual element in the image.\n\n"
                    f"TARGET: {step.target_desc}\n\n"
                    "Copy this line EXACTLY, replacing each ? with a coordinate integer in [0, 1000]:\n"
                    f"\"{step.id}: {step.target_desc}\" -> [?, ?, ?, ?]\n\n"
                    "RULES:\n"
                    "1. Replace ? with integers in [0, 1000] (normalized to image dimensions).\n"
                    "2. Box the ENTIRE region described — not a tight corner, but the full area.\n"
                    '3. If NOT visible, replace all four ? with -1.\n'
                    "4. The LAST line of your response must be the completed coordinate line. Nothing else after.\n"
                    "5. The line must be character-for-character IDENTICAL except ? → numbers."
                )

                compact_plan = (
                    f"You are solving: <q>{question}</q>\n"
                    "Create an atomic execution plan."
                )
                history = [
                    {'role': 'user', 'content': [
                        {'type': 'text', 'value': compact_plan},
                        {'type': 'image', 'value': image_path},
                    ]},
                    {'role': 'assistant', 'content': [{'type': 'text', 'value': plan_text[:2000]}]},
                    {'role': 'user', 'content': [{'type': 'text', 'value': grounding_prompt}]},
                ]

                ret, grounding_text, resp = self.grounder.generate_inner(history)
                _t_elapsed = _time.time() - _t_start
                _log_diag(f'[V2 Inspect] Grounding done in {_t_elapsed:.0f}s | {len(grounding_text) if grounding_text else 0} chars')

                if ret != 0 or not grounding_text:
                    evidence.add(Evidence(
                        step_id=step.id, label=step.target_desc,
                        value=None, confidence=0.0, raw_text='GROUNDING_FAILED',
                    ))
                    states[step.id] = StepState.FAILED
                    continue

                # Parse coordinates
                labeled_pattern = r'"([^"]*)"\s*->\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
                matches = re.findall(labeled_pattern, grounding_text)

                if not matches:
                    bare_pattern = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
                    bare_matches = re.findall(bare_pattern, grounding_text)
                    if bare_matches:
                        bm = bare_matches[0]
                        label = step.target_desc[:60]
                        matches = [(f"{step.id}: {label}",) + bm]

                if not matches:
                    _log_diag(f'[V2 Inspect] No coordinates found for {step.id}')
                    evidence.add(Evidence(
                        step_id=step.id, label=step.target_desc,
                        value=None, confidence=0.0, raw_text='NO_COORDINATES',
                    ))
                    states[step.id] = StepState.FAILED
                    continue

                m = matches[0]
                bbox_norm = [int(m[1]), int(m[2]), int(m[3]), int(m[4])]
                raw_label = m[0].strip()
                if ':' in raw_label:
                    label = raw_label.split(':', 1)[1].strip()
                else:
                    label = raw_label

                # Validate
                if not self._validate_grounding(bbox_norm, orig_w, orig_h):
                    _log_diag(f'[V2 Inspect] Grounding validation FAILED for {step.id}: {bbox_norm}')
                    evidence.add(Evidence(
                        step_id=step.id, label=label,
                        value=None, confidence=0.1, bbox_norm=bbox_norm,
                        raw_text=f'VALIDATION_FAILED: {raw_label}',
                    ))
                    states[step.id] = StepState.FAILED
                    continue

                # Crop with generous padding
                total_read_calls += 1
                p_bbox = [
                    int((bbox_norm[0] / 1000.0) * orig_w),
                    int((bbox_norm[1] / 1000.0) * orig_h),
                    int((bbox_norm[2] / 1000.0) * orig_w),
                    int((bbox_norm[3] / 1000.0) * orig_h),
                ]
                # Manual crop with inspect padding/scale (different from default)
                from PIL import Image as PILImage
                with PILImage.open(image_path).convert('RGB') as img:
                    xmin, ymin, xmax, ymax = p_bbox
                    if xmin > xmax:
                        xmin, xmax = xmax, xmin
                    if ymin > ymax:
                        ymin, ymax = ymax, ymin
                    if xmin == xmax:
                        xmax += 1
                    if ymin == ymax:
                        ymax += 1
                    c_xmin = max(0, xmin - self.inspect_padding)
                    c_ymin = max(0, ymin - self.inspect_padding)
                    c_xmax = min(img.width, xmax + self.inspect_padding)
                    c_ymax = min(img.height, ymax + self.inspect_padding)
                    cropped = img.crop((c_xmin, c_ymin, c_xmax, c_ymax))
                    if self.inspect_scale != 1.0:
                        new_w = int(cropped.width * self.inspect_scale)
                        new_h = int(cropped.height * self.inspect_scale)
                        cropped = cropped.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
                    w, h = cropped.size
                    _max = max(w, h)
                    if _max > self.max_crop_edge:
                        ratio = self.max_crop_edge / _max
                        cropped = cropped.resize((int(w * ratio), int(h * ratio)), PILImage.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    cropped.save(buffer, format='JPEG', quality=95)
                    cropped_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # Visual analysis question
                inspect_prompt = (
                    "/no_think\n"
                    f"Analyze this image region and answer the question.\n\n"
                    f"QUESTION: {step.description}\n"
                    f"TARGET: {step.target_desc}\n"
                    f"Expected answer type: {step.expected_type}\n\n"
                    "CRITICAL RULES:\n"
                    "1. Output ONLY the answer. NOTHING else. No reasoning, no explanation.\n"
                    "2. Your ENTIRE response must be a single line.\n"
                    "3. For counts: just the number (e.g., '3'). For booleans: YES or NO.\n"
                    "4. For text values: just the text.\n"
                    "5. If unclear: output UNCLEAR. If not found: output NOT_FOUND."
                )

                inspect_content = [
                    {'type': 'text', 'value': inspect_prompt},
                    {'type': 'image_base64', 'value': cropped_b64},
                ]
                inspect_history = [{'role': 'user', 'content': inspect_content}]

                ret, inspect_text, resp = super().generate_inner(inspect_history)
                _t_elapsed = _time.time() - _t_start
                _log_diag(f'[V2 Inspect] Done in {_t_elapsed:.0f}s | {len(inspect_text) if inspect_text else 0} chars')

                if ret != 0 or not inspect_text:
                    evidence.add(Evidence(
                        step_id=step.id, label=label,
                        value=None, confidence=0.0, raw_text='INSPECT_FAILED',
                    ))
                    states[step.id] = StepState.FAILED
                    continue

                value_str = inspect_text.strip()
                extracted = extract_final_value(value_str)
                if extracted:
                    value_str = extracted

                cleaned_val, was_cleaned = clean_value(value_str)
                if cleaned_val is None:
                    conf = 0.3
                elif was_cleaned:
                    conf = 0.8
                else:
                    conf = 1.0

                evidence.add(Evidence(
                    step_id=step.id,
                    label=label,
                    value=cleaned_val,
                    confidence=conf,
                    bbox_norm=bbox_norm,
                    raw_text=value_str,
                    unit=step.expected_unit,
                ))
                states[step.id] = StepState.DONE
                _log_diag(f'[V2 Inspect] {step.id} DONE: [{label}] = {cleaned_val}')

            elif step.type == 'compute':
                result = self._execute_compute(step, evidence)
                if result is not None:
                    evidence.add(Evidence(
                        step_id=step.id, label=step.description,
                        value=result, confidence=1.0, raw_text=str(result),
                    ))
                    states[step.id] = StepState.DONE
                    _log_diag(f'[V2 Loop] {step.id} DONE: {step.description} = {result}')
                else:
                    states[step.id] = StepState.FAILED
                    _log_diag(f'[V2 Loop] {step.id} FAILED: could not compute')

            elif step.type == 'verify':
                states[step.id] = StepState.DONE

            # 3. After each step, check if all done and sufficiency
            pending_after = [s for s in all_steps if states[s.id] == StepState.PENDING]
            if not pending_after:
                is_sufficient, missing = self._check_sufficiency(evidence, all_steps, question)
                if not is_sufficient and iteration < self.max_iterations:
                    _log_diag(f'[V2 Loop] Evidence insufficient, re-planning...')
                    new_steps = self._replan(evidence, missing, image_path, question, orig_w, orig_h)
                    if new_steps:
                        for ns in new_steps:
                            all_steps.append(ns)
                            states[ns.id] = StepState.PENDING
                        _log_diag(f'[V2 Loop] Added {len(new_steps)} new steps from re-plan')
                    else:
                        _log_diag('[V2 Loop] Re-plan returned no new steps, proceeding')
                        break
                else:
                    break

        done_count = sum(1 for s in states.values() if s == StepState.DONE)
        failed_count = sum(1 for s in states.values() if s == StepState.FAILED)
        pending_count = sum(1 for s in states.values() if s == StepState.PENDING)
        _log_diag(f'[V2 Loop] Complete: {done_count} done, {failed_count} failed, '
                  f'{pending_count} pending | {total_grounding_calls} grounding calls, '
                  f'{total_read_calls} read calls | {len(evidence)} evidence items')

        return evidence


    def final_reasoning(self, evidence: EvidenceStore, image_path: str,
                        question: str, orig_w: int, orig_h: int) -> str:
        """Phase 3: Generate the final answer using collected evidence + original image."""
        _t_start = _time.time()
        _log_diag('[V2 Phase3] Final Reasoning start')

        evidence_summary = evidence.compress_for_context(max_items=20)
        read_evidence = evidence.get_read_evidence()

        # Build prompt
        prompt_s3 = (
            "/no_think\n"
            "You are solving a visual reasoning problem. Below is extracted evidence and the original image.\n\n"
            "=== COLLECTED EVIDENCE ===\n"
            f"{evidence_summary}\n\n"
            "=== ORIGINAL QUESTION ===\n"
            f"<q>{question}</q>\n\n"
            "=== INSTRUCTIONS ===\n"
            "1. The evidence contains raw data extracted from the image. TRUST the values.\n"
            "2. You MAY use the original image to FILTER, CATEGORIZE, or verify evidence items.\n"
            "3. Be concise — show only the calculation steps, no extra reasoning. Under 200 words.\n"
            "4. At the very END of your response, enclose the final answer in "
            "<a> and </a> tags like this: <a>your answer</a>.\n"
            "5. Do NOT re-read values already in evidence. Do NOT include thinking traces."
        )

        s3_content = [{'type': 'text', 'value': prompt_s3}]

        # Always attach original image (scaled down) for global context only
        orig_b64 = self._resize_and_encode_original(image_path, max_edge=1024)
        s3_content.append({'type': 'image_base64', 'value': orig_b64})

        # Build history with planning context (neutralized)
        s3_history = [
            {'role': 'user', 'content': [
                {'type': 'text', 'value': f"You are solving: <q>{question}</q>"},
            ]},
            {'role': 'user', 'content': s3_content},
        ]

        # Disable extra_body to avoid min_pixels constraints on crops
        saved_extra_body = self.extra_body
        self.extra_body = None
        try:
            ret, answer, resp = super().generate_inner(s3_history)
            _t_elapsed = _time.time() - _t_start
            _log_diag(f'[V2 Phase3] Done in {_t_elapsed:.0f}s | {len(answer)} chars')
            return answer
        finally:
            self.extra_body = saved_extra_body

    # ============================================================
    # Main Entry Point: generate_inner
    # ============================================================

    def generate_inner(self, inputs, **kwargs):
        """Main entry point called by the VLMEvalKit framework.

        Orchestrates all three phases: Planning → Execute → Reason.
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
            _log_diag('[V2] No image in inputs, falling back to parent')
            return super().generate_inner(inputs, **kwargs)

        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        # ---- Phase 1: Structured Planning ----
        steps, plan_text = self.structured_planning(image_path, original_question)

        if not steps:
            _log_diag('[V2] Planning produced no steps, falling back to direct reasoning')
            # Fallback: direct question answering
            fallback = [
                {'type': 'text', 'value': f"Answer this question step by step: <q>{original_question}</q>. "
                                          "Enclose final answer in <a>...</a> tags."},
                {'type': 'image', 'value': image_path},
            ]
            ret, answer, resp = super().generate_inner(fallback)
            _t_total = _time.time() - _t_total_start
            _log_diag(f'[V2] Fallback completed in {_t_total:.0f}s')
            return ret, answer, resp

        # ---- Phase 2: Iterative Execute-Verify Loop ----
        evidence = self.iterative_execute(
            steps, image_path, original_question, orig_w, orig_h, plan_text
        )

        # ---- Phase 3: Final Reasoning ----
        answer = self.final_reasoning(
            evidence, image_path, original_question, orig_w, orig_h
        )

        _t_total = _time.time() - _t_total_start
        n_read_evidence = len(evidence.get_read_evidence())
        _log_diag(f'[V2] Total: {_t_total:.0f}s | {len(steps)} plan steps | '
                  f'{n_read_evidence} read evidence | {len(evidence)} total evidence')

        # Return in the framework-expected format
        # answer is the final text; response is None (not an API call directly)
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

        # Try FINAL_ANSWER: marker
        match = re.search(r'FINAL_ANSWER\s*:\s*(.+?)(?:\n|$)', text)
        if match:
            return match.group(1).strip()

        # Fallback: last non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[-1] if lines else ""
