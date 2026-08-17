"""
OCR-Agent V5 Grounding-Understanding: Planning → Executor (DAG-based) → Reasoning Pipeline
===========================================================================================
Inherits OCRAgentV5 with a modified ground-understand step that splits into:
  Phase 1 - Grounding:  locate + crop regions of interest from the original image
  Phase 2 - Understanding: feed cropped image(s) + grounding description + step goal
             to a second model call → one-sentence JSON answer

All other stages (Planning, compute, logical, transcribe, Reasoning) are identical to V5.

Key difference from OCRAgentV5:
  - Does NOT support skip_locate (always runs full Grounding → Understanding)
  - Grounding prompt emphasizes descriptive purpose text for downstream Understanding
  - Understanding prompt explicitly references grounding descriptions + step goal
  - Enforces uniform JSON output format across all phases
"""
import os
import io
import re
import json
import base64
import copy
from PIL import Image, ImageDraw, ImageFont
from vlmeval.api.gpt import OpenAIWrapper, _log_diag
from vlmeval.vlm.ocr_agent_v5 import (
    OCRAgentV5,
    _Base64Agent,
    PLANNING_SCHEMA,
    LOCATE_SCHEMA,
    COMPUTE_SCHEMA,
    LOGICAL_SCHEMA,
    TRANSCRIBE_SCHEMA,
    REASONING_SCHEMA,
    PLANNING_PROMPT,
    TRANSCRIBE_PROMPT,
    COMPUTE_PROMPT,
    LOGICAL_PROMPT,
    OVERVIEW_PROMPT,
    REASONING_PROMPT,
    MAX_ATTEMPTS,
)


# ============================================================================
# Grounding-Understanding specific JSON Schemas
# ============================================================================

# Grounding uses the same LOCATE_SCHEMA but with a richer "purpose" field
GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "bbox_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "purpose": {
                        "type": "string",
                        "description": "A concise description of what this region contains — this description will be forwarded to the Understanding phase, so make it specific and self-contained"
                    },
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

GROUNDING_UNDERSTAND_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {
            "type": "string",
            "description": "A single self-contained descriptive sentence answering the step's target question, based on the cropped image(s) and grounding descriptions"
        }
    },
    "required": ["result"]
}


# ============================================================================
# Grounding-Understanding specific Prompts
# ============================================================================

GROUNDING_PROMPT = """You are a precise visual localizer — a "pointer", NOT a reader. Your ONLY job is to output bounding box coordinates for regions in the image. You do NOT read values, analyze data, or solve problems.

## CRITICAL — WHAT YOU MUST NOT DO
1. **DO NOT read or analyze any data values.** You are not here to understand numbers, compare quantities, or figure out what the chart means.
2. **DO NOT try to understand the chart structure in depth.** If the task says "far right side of the bar", just locate the visual region at the right end of that bar — don't analyze what the number represents.
3. **DO NOT list or enumerate all items in the image.** Only locate what the task asks for.
4. **THINK IN UNDER 3 SENTENCES.** Then output coordinates immediately. Long analysis wastes time and causes failure.

## Instructions
1. Read the step description — understand WHAT visual element to point at.
2. Glance at the image to find WHERE that element is. Do NOT read its value.
3. Output TIGHT bounding boxes. Include supporting elements (axis, legend) ONLY if needed to identify the target.
4. Write a **descriptive purpose** for each region group — this will be forwarded to the next phase (Understanding) so the reader knows what to look for.

## Coordinate System
- [x1, y1, x2, y2] in per-mille [0, 1000], (0,0)=top-left.
- Integers only, x1<x2, y1<y2, TIGHT around target.

## Output Format
{"bbox_groups": [{"group_id": "...", "purpose": "A concise description of what this region shows — forwarded to the Understanding phase", "bboxes": [[x1,y1,x2,y2], ...]}]}

## CRITICAL — Purpose Field
The "purpose" field MUST be a descriptive sentence like:
- "The blue bar labeled 'Q1 Sales' showing a value near the top of the y-axis range"
- "The y-axis with numerical scale from 0 to 500"
- "The legend showing color-to-category mappings"

Do NOT write vague purposes like "a bar" or "some text" — be specific so the Understanding phase can use this context."""

GROUNDING_UNDERSTAND_PROMPT = """You are a careful visual reader performing the **Understanding** phase. You will be shown:
1. Cropped image regions (from the Grounding phase)
2. A description of what each crop contains (the "purpose" from the Grounding phase)
3. A specific target question describing what information to extract

## Your Job
Synthesize information from ALL cropped regions together with their grounding descriptions to answer the target question.

## Instructions
1. Read each crop's "purpose" label to understand what the crop contains.
2. Look at each cropped image carefully — read text, numbers, scales, positions, colors, and any other visual information.
3. Cross-reference between crops: e.g., use the axis crop to interpret the value shown in the bar crop; use the legend crop to identify which category the bar represents.
4. Produce a **single self-contained descriptive sentence** that directly answers the target question.

## Output Format
You MUST output a single JSON object:
{"result": "your one-sentence answer here"}

## CRITICAL — One Complete Sentence Required
Your "result" MUST be exactly **one self-contained descriptive sentence** that a person can understand without seeing the original image or the task description. Never output a bare number, single word, or bullet list.

GOOD examples:
- "The sales value for Q1 is 42.5 (in units of 10,000 RMB)"
- "The label for the blue bar in the legend is 'Q1 Sales'"
- "The y-axis ranges from 0 to 500, with major gridlines every 100 units"
- "The price of '纯牛肉丸汤' on the menu is 20 yuan"
- "All four bars (Q1: 42.5, Q2: 38.2, Q3: 55.1, Q4: 60.3) in units of 10,000 RMB"

BAD examples (FORBIDDEN):
- "42.5"           ← bare number, no meaning to downstream steps
- "Q1 Sales"       ← label without context
- "0 to 500"       ← range without specifying what it refers to
- "Q1: 42.5, Q2: 38.2"  ← fragment, not a complete sentence"""


# ============================================================================
# Main Agent Class
# ============================================================================

class OCRAgentV5Grounding(OCRAgentV5):
    """V5 OCR Agent with Grounding-Understanding pipeline for ground-understand steps.

    Inherits all V5 logic (Planning, compute, logical, transcribe, Reasoning).
    Only overrides the ground-understand executor to use a dedicated two-phase
    Grounding → Understanding pipeline with specialized prompts and consistent
    JSON output format.

    Key behavioral differences from OCRAgentV5:
      - skip_locate is always False (Grounding phase is required)
      - Grounding prompt emphasizes descriptive purpose text
      - Understanding prompt takes cropped images + grounding descriptions + step goal
      - Output is always a one-sentence self-contained JSON result
      - No cross-validation on the understanding call (single pass, grounded on crops)
    """

    def __init__(self, **kwargs):
        # Force skip_locate to False — this variant always uses Grounding
        kwargs['skip_locate'] = False
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Ground-Understand: two-phase Grounding → Understanding
    # ------------------------------------------------------------------

    def _execute_ground_understand(self, step, context, image_path, orig_w, orig_h,
                                    question='', use_cv=True):
        """Execute ground-understand as a two-phase pipeline:
        Phase 1 (Grounding):  Locate → Crop relevant regions
        Phase 2 (Understanding): Feed cropped images + grounding descriptions
                                 + step goal → one-sentence JSON answer

        This overrides the parent's method. skip_locate fast-path is never used.
        """
        description = step.get('description', '')
        step_id = step.get('id', 'unknown')

        # ================================================================
        # Phase 1: Grounding — Locate regions + Crop
        # ================================================================
        _log_diag(f'[V5-Grounding] {step_id}: Phase 1 — Grounding started')

        locate_user = (
            f"Step Goal (target to extract): {description}\n\n"
            f"{context}\n"
            "Point to the regions needed. Do NOT read values or analyze data. "
            "Write descriptive purpose labels for each region — these will be "
            "forwarded to the Understanding phase. "
            "Think in 1-2 sentences, then output the JSON immediately."
        )

        locate_messages = [
            {'type': 'text', 'value': locate_user},
            {'type': 'image', 'value': image_path},
        ]

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = GROUNDING_PROMPT
        try:
            ret, locate_text, _ = self._api_call(
                locate_messages,
                schema=GROUNDING_SCHEMA,
                with_image_processor=True,
                temperature=self.temp_locate,
            )
        finally:
            self.agent.system_prompt = saved_sp

        locate_obj, locate_err = self._parse_json(locate_text, GROUNDING_SCHEMA)
        _log_diag(f'[V5-Grounding] {step_id}: err={locate_err}, '
                  f'obj_keys={list(locate_obj.keys()) if locate_obj else None}')

        if locate_obj is None or not locate_obj.get('bbox_groups'):
            return {
                'status': 'error',
                'type': 'ground-understand',
                'result': '',
                'error': f'Grounding failed: {locate_err or "empty bbox_groups"}',
                'failed_phase': 'locate',
            }

        bbox_groups = locate_obj['bbox_groups']

        # Draw debug bboxes
        self._draw_bboxes_on_image(
            image_path, bbox_groups, step_id, orig_w, orig_h
        )

        # Compute physical crops (same logic as parent)
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
                'error': 'Grounding produced no valid bounding boxes',
                'failed_phase': 'locate',
                'bbox_groups': bbox_groups,
            }

        # ================================================================
        # Phase 2: Understanding — Crops + Grounding descriptions + Goal
        # ================================================================
        _log_diag(f'[V5-Understanding] {step_id}: Phase 2 — Understanding started '
                  f'({len(all_crops)} crops)')

        # Build understanding prompt with grounding descriptions
        understand_user_content = [
            {'type': 'text', 'value': (
                f"## Step Goal (target to extract)\n{description}\n\n"
                f"{context}\n"
                f"## Grounding Results\n"
                f"The Grounding phase located the following regions. "
                f"Each cropped image below is labeled with its purpose. "
                f"Read and synthesize information from ALL crops to answer "
                f"the Step Goal above.\n"
            )}
        ]

        current_group = None
        for group_id, purpose, bbox_idx, phys_bbox in all_crops:
            if group_id != current_group:
                current_group = group_id
                understand_user_content.append({
                    'type': 'text',
                    'value': (f'\n--- Region Group: {group_id} ---\n'
                              f'Grounding description (what this crop shows): {purpose}')
                })

            understand_user_content.append({
                'type': 'text',
                'value': f'[Crop {bbox_idx+1} of group "{group_id}"]',
            })

            b64_str = self._crop_and_encode(image_path, phys_bbox)
            understand_user_content.append({
                'type': 'image_base64',
                'value': b64_str,
            })

        # Final instruction for one-sentence output
        understand_user_content.append({
            'type': 'text',
            'value': (
                '\n## Output Instruction\n'
                'Synthesize information from ALL cropped images and their grounding '
                'descriptions. Output exactly ONE self-contained descriptive sentence '
                'that answers the Step Goal. Output ONLY the JSON object.'
            )
        })

        saved_sp = self.agent.system_prompt
        self.agent.system_prompt = GROUNDING_UNDERSTAND_PROMPT
        try:
            ret, understand_text, _ = self._api_call(
                understand_user_content,
                schema=GROUNDING_UNDERSTAND_SCHEMA,
                with_image_processor=False,
                temperature=self.temp_understand,
            )
        finally:
            self.agent.system_prompt = saved_sp

        understand_obj, understand_err = self._parse_json(
            understand_text, GROUNDING_UNDERSTAND_SCHEMA
        )
        _log_diag(f'[V5-Understanding] {step_id}: err={understand_err}')

        if understand_obj is None:
            return {
                'status': 'error',
                'type': 'ground-understand',
                'result': '',
                'error': f'Understanding failed: {understand_err}',
                'failed_phase': 'understand',
                'bbox_groups': bbox_groups,
            }

        result_text = understand_obj.get('result', '')
        _log_diag(f'[V5-Understanding] {step_id}: OK — {result_text[:120]}')

        return {
            'status': 'ok',
            'type': 'ground-understand',
            'result': result_text,
            'bbox_groups': bbox_groups,
        }
