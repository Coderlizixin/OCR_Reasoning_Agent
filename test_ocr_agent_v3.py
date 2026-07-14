"""
Test script for OCRAgent V3 — Sequential Grounding → Understanding Pipeline.

Architecture:
  Phase 1: Structured Planning (JSON schema, atomic steps, DAG with $placeholders)
  Phase 2: Sequential Execution Loop
    - ground_and_understand: Grounding (multi-bbox) → Understanding (natural language answer)
    - compute: LLM computation
    - logical: LLM for filtering/judgment
    - $step_id placeholders resolved at runtime from dependency evidence
  Phase 3: Final Reasoning with complete evidence chain

Key design decisions:
  - Grounding and Understanding are SEPARATED (2 API calls per visual step)
  - Multiple bboxes allowed per step (e.g., legend + axis + data point for one chart reading)
  - Dependency context is PURE TEXT (no crop images passed between steps)
  - Understanding outputs natural language, passed as-is to dependent steps
  - All models use 9B
  - Placeholder mechanism: $s2 in description → resolved at runtime

Run: python test_ocr_agent_v3.py
"""

import base64
import io
import json
import os
import re
import ast

import time as _time
from PIL import Image, ImageDraw
from openai import OpenAI


# ==========================================
# 0. Configuration
# ==========================================
# LOCAL_IMAGE_PATH = "test_case/test_images/image14.jpg"
# QUESTION = "What is the total price of the foods whose names end with the letter 'r'?"
# LOCAL_IMAGE_PATH = "test_case/test_images/image0.jpg"
# QUESTION = "What is the average number of the genres of movies made by directors who have worked in drama genres?"
# LOCAL_IMAGE_PATH = "test_case/test_images/image4.jpg"
# QUESTION = "How many are somewhat confident that Donald Trump can manage the executive branch effectively?"
# QUESTION = "What does the \"45\" of \"manage the executive branch effectively?\" mean?"
# LOCAL_IMAGE_PATH = "test_case/test_images/image22.jpg"
# QUESTION = "曲线的最高点比收盘价高多少？"
LOCAL_IMAGE_PATH = "test_case/test_images/image13.jpg"
QUESTION = "Do consumers aged 40-45 buy the most pens?"

# Model endpoints (all 9B)
API_BASE_URL = "http://10.3.112.34:22630/v1"
MODEL_NAME = "Qwen3.5-9B"

# Crop parameters
PADDING = 50           # pixel padding for tight crops
SCALE_FACTOR = 2.5     # magnification
MAX_CROP_EDGE = None   # no compression on crop images
MAX_CROPS = 44          # max crop images; overflow → text-only (label + bbox)

# Execution limits
MAX_ITERATIONS = 30    # safety cap for sequential loop

# Output
OUTPUT_DIR = "test_case/ocr_agent_v3_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 1. Helper Functions
# ==========================================

def log(msg):
    """Timestamped log."""
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}")


def encode_full_resolution(image_path):
    """Read local image at FULL resolution — no scaling, no compression.
    Used for Planning and Grounding phases where the model needs to see all details."""
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
        buffer = io.BytesIO()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), orig_w, orig_h


def resize_and_encode(image_path, max_size=2048):
    """Read local image, scale down for global context, return (base64, orig_w, orig_h)."""
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))
            print(f"[resize] Image scaled to: {img.width}x{img.height}")
        buffer = io.BytesIO()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), orig_w, orig_h


def crop_and_encode(image_path, physical_bbox, padding=PADDING,
                    scale_factor=SCALE_FACTOR, max_edge=MAX_CROP_EDGE):
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

        c_xmin = max(0, xmin - padding)
        c_ymin = max(0, ymin - padding)
        c_xmax = min(img.width, xmax + padding)
        c_ymax = min(img.height, ymax + padding)

        cropped = img.crop((c_xmin, c_ymin, c_xmax, c_ymax))

        if scale_factor != 1.0:
            new_w = int(cropped.width * scale_factor)
            new_h = int(cropped.height * scale_factor)
            cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

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


def encode_pil_to_base64(pil_image, max_edge=1024):
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


def build_image_content(b64_string):
    return {
        'type': 'image_url',
        'image_url': {'url': f"data:image/jpeg;base64,{b64_string}"}
    }


def build_text_content(text):
    return {'type': 'text', 'text': text}


def parse_json_robust(text):
    """Robustly extract JSON from model output (handles thinking traces)."""
    if not text:
        return None

    # Strip thinking/response tags
    for tag in ['<think>', '<|end_of_thinking|>', ' response']:
        for end_tag in ['</think>', '</think>', '</think>']:
            if end_tag in text:
                parts = text.rsplit(end_tag, 1)
                text = parts[-1]
            if tag in text and end_tag in text:
                text = re.sub(re.escape(tag) + r'.*?' + re.escape(end_tag),
                             '', text, flags=re.DOTALL)

    # Strip markdown code fences and prefixes
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


def clean_value(value_str):
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


def extract_final_value(text):
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


# ==========================================
# 3. Initialize OpenAI Client
# ==========================================

client = OpenAI(api_key="EMPTY", base_url=API_BASE_URL)
log(f"Model: {MODEL_NAME} @ {API_BASE_URL}")

# ==========================================
# 4. Preprocess Image
# ==========================================

# Full-resolution encoding for Planning and Grounding (no compression)
base64_image_full, orig_w, orig_h = encode_full_resolution(LOCAL_IMAGE_PATH)
# Downscaled encoding for global context in Understanding and Final Reasoning
base64_image_thumb, _, _ = resize_and_encode(LOCAL_IMAGE_PATH, max_size=2048)

log(f"Image: {LOCAL_IMAGE_PATH} ({orig_w}x{orig_h})")
log(f"  Full-res base64: {len(base64_image_full)} chars")
log(f"  Thumb base64: {len(base64_image_thumb)} chars")
log(f"Question: {QUESTION}")

original_img = Image.open(LOCAL_IMAGE_PATH).convert('RGB')
draw = ImageDraw.Draw(original_img)


# ==========================================
# 5. Phase 1: Structured Planning
# ==========================================
_t1_start = _time.time()
log("=" * 60)
log("PHASE 1: Structured Planning (JSON Plan)")

PLANNING_PROMPT = (
    f"Create a JSON execution plan to answer this question: <q>{QUESTION}</q>\n\n"
    "=== ROLE ===\n"
    "You are a PLANNER. Decompose the question into SMALL, ATOMIC steps.\n"
    "Each step does ONE tiny thing. Output ONLY the JSON plan.\n\n"
    "=== STEP TYPES ===\n"
    '- "<ground-understand>": Read ONE kind of element from ONE contiguous region and understand its meaning. Needs image.\n'
    '- "<compute>": ONE arithmetic operation (sum, avg, diff, etc.). No image.\n'
    '- "<logical>": ONE filter, sort, group, or match operation. No image.\n\n'
    "=== WHAT TO SKIP (IMPORTANT) ===\n"
    "Do NOT create steps for:\n"
    "- Reading the chart title, subtitle, or captions (irrelevant metadata)\n"
    "- Reading the legend (unless the question asks about categories shown IN the legend)\n"
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
    "R1. FILTERING IS A SEPARATE logical STEP: Never put conditions into <ground-understand>.\n"
    "R2. ONE SEMANTIC UNIT PER STEP: One column, one row, one axis, one label "
    "group, one value. If a description contains \"and\" or \"with their\", "
    "SPLIT IT.\n"
    "R3. USE $step_id PLACEHOLDERS: \"filter $s1 where price > $s2\".\n\n"
    "=== CONCRETE EXAMPLE ===\n"
    'Question: "For the countries whose steel export volume is higher than the average, what is the total amount of their oil imports?"\n'
    "Structure: type=chart, 2 bars [steel export, oil import] for each country, 5 countries\n\n"
    "GOOD plan (5 steps):\n"
    '  s1 (<ground-understand>): "Read steel export of all countries."\n'
    '  s2 (<compute>): "Compute average of steel export."\n'
    '  s3 (<logical>): "Select the countries whose steel export are higher than the average $s2."\n'
    '  s4 (<ground-understand>): "Read oil import of the countries in $s3."\n'
    '  s5 (<compute>): "Compute the total amount of oil imports in countries $s4."\n'
    "BAD plan (3 steps — TOO COARSE):\n"
    '  s1: "Read all employee data from the chart."\n'
    '  s2: "Select the countries whose steel export are higher than the average."\n'
    '  s3: "Compute the total amount of oil imports in countries."\n\n'
    "=== JSON SCHEMA ===\n"
    '{"steps": [\n'
    '  {"id": "s1", "type": "<ground-understand>", '
    '"description": "Read all X from region Y. Output as list.", '
    '"expected_type": "list[str]", "depends_on": []},\n'
    '  ...\n'
    ']}\n\n'
    "OUTPUT ONLY THE JSON."
    # "OUTPUT ONLY THE JSON. No thinking, no markdown, no explanation."
)

plan_response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.0,
    max_tokens=12288,
    extra_body={
        # "enable_thinking": False,
        # "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {
            "images_kwargs": {
                "size": {
                    "longest_edge": 8192 * 32 * 32,
                    "shortest_edge": 4096 * 32 * 32,
                }
            }
        }
    },
    messages=[{
        'role': 'user',
        'content': [build_text_content(PLANNING_PROMPT), build_image_content(base64_image_full)]
    }]
)

plan_text = plan_response.choices[0].message.content
_t1_elapsed = _time.time() - _t1_start

plan_json = parse_json_robust(plan_text)
n_steps = len(plan_json.get('steps', [])) if plan_json else 0
log(f"Phase 1 done ({_t1_elapsed:.0f}s) | {n_steps} steps")

if plan_json is None:
    log("PLANNING FAILED — falling back to direct reasoning with original image")
    print(f"[Raw plan output]: {plan_text[:500]}")
    FALLBACK = True
    planning_success = False
    plan_steps = []
else:
    FALLBACK = False
    planning_success = True

if not FALLBACK:
    print(f"\n{'─' * 40}\n[Plan JSON]:\n{json.dumps(plan_json, ensure_ascii=False, indent=2)}\n{'─' * 40}\n")

# Parse plan steps (skip if fallback)
if not FALLBACK:
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


# ==========================================
# 6. Placeholder Resolution
# ==========================================

def resolve_placeholders(text, evidence):
    """Replace $step_id placeholders with actual values from evidence.

    Supports:
      $s2       → the value of step s2
      $s2.xxx   → sub-field (reserved for future use)
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
            # $s2 → the step's primary value/answer
            val = ev.get('value')
            if val is None:
                return f"[UNRESOLVED:{ref}]"
            # For complex data (lists), just use the raw answer text
            if isinstance(val, (list, dict)):
                return json.dumps(val, ensure_ascii=False)
            return str(val)
        else:
            # $s2.something → sub-field
            sub = parts[1]
            # Try sub-fields dict, then raw reference
            sub_fields = ev.get('sub_fields', {})
            if sub in sub_fields:
                return str(sub_fields[sub])
            return f"[UNRESOLVED:{ref}]"

    return re.sub(r'\$(\w+(?:\.\w+)*)', replace_ref, text)




# ==========================================
# 7. Phase 2: Sequential Execution Loop
# ==========================================
_t2_start = _time.time()
log("=" * 60)
log("PHASE 2: Sequential Execution Loop")

# Track state
states = {}     # step_id → 'PENDING' | 'EXECUTING' | 'DONE' | 'FAILED'
evidence = {}   # step_id → {label, value, raw_answer, bboxes[], sub_fields}
all_steps = list(plan_steps)

for s in all_steps:
    states[s['id']] = 'PENDING'

iteration = 0
total_grounding_calls = 0
total_understanding_calls = 0
total_compute_llm_calls = 0
total_logical_llm_calls = 0

while iteration < MAX_ITERATIONS:
    iteration += 1

    # ---- Select ONE ready step ----
    ready = []
    for s in all_steps:
        if states[s['id']] != 'PENDING':
            continue
        if all(states.get(dep) == 'DONE' for dep in s['depends_on']):
            ready.append(s)

    if not ready:
        pending = [s for s in all_steps if states[s['id']] == 'PENDING']
        if not pending:
            log("All steps completed!")
            break
        # Stuck — try unblocking steps with no deps
        ready = [s for s in pending if not s['depends_on']]
        if not ready:
            log(f"WARNING: {len(pending)} steps stuck with unmet dependencies")
            break

    ready.sort(key=lambda s: (s.get('priority', 99), s['id']))
    step = ready[0]
    sid = step['id']
    log(f"--- Iteration {iteration}/{MAX_ITERATIONS} | Step {sid} ({step['type']}) ---")

    states[sid] = 'EXECUTING'

    # ---- Resolve placeholders ----
    resolved_description = resolve_placeholders(step.get('description', ''), evidence)

    # ---- Build dependency context (pure text) ----
    dep_context_lines = []
    for dep_id in step['depends_on']:
        ev = evidence.get(dep_id, {})
        if ev.get('raw_answer'):
            dep_context_lines.append(
                f"[Result of Step {dep_id}]: {ev['raw_answer']}"
            )
        elif ev.get('value') is not None:
            dep_context_lines.append(
                f"[Result of Step {dep_id}]: {ev['value']}"
            )
    dep_context = "\n".join(dep_context_lines) if dep_context_lines else "(No previous results)"

    # ================================================================
    # Execute: ground_and_understand
    # ================================================================
    if step['type'] == '<ground-understand>':
        # ----------------------------------------------------------
        # Step 2a: Grounding — locate visual elements
        # ----------------------------------------------------------
        _t_ground = _time.time()
        total_grounding_calls += 1

        grounding_prompt = (
            "You are a visual grounding assistant. Locate visual elements with "
            "SEMANTIC labels AND coordinates. Besides, give the answer of your target.\n\n"
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
            # "=== ANCHOR-BASED TARGETING ===\n"                                                                                                                                                                                                                                                         
            # "Refine them to TIGHT boxes around the actual content.\n
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
            "1. MANDATORY: Every line must have a semantic label + arrow + coords.\n"
            "2. Box the TIGHTEST rectangle. Normalize to [0,1000] where (0,0) is "
            "top-left and (1000,1000) is bottom-right. All values must be integers..\n"
            "3. Each box = ONE element. Don't group unrelated items.\n"
            "4. If a requested NUMBER or TEXT is not printed inside an element,\n"
            "   box the element itself (the bar, the cell, the region) and\n"
            "   label it clearly. NEVER output text-only explanations without\n"
            "   coordinates.\n"
            "Output ONLY labeled coordinate lines. And give the answer of your "
            "target, e.g., 'The price of water is $2.' 'The legent Very is "
            "colored in dark gold.'\n"
        )

        ground_response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=6144,
            extra_body={
                "mm_processor_kwargs": {
                    "images_kwargs": {
                        "size": {
                            "longest_edge": 2048 * 32 * 32,
                            "shortest_edge": 64 * 32 * 32,
                        }
                    }
                }
            },
            messages=[{
                'role': 'user',
                'content': [
                    build_text_content(grounding_prompt),
                    build_image_content(base64_image_full),
                ]
            }]
        )

        grounding_text = ground_response.choices[0].message.content
        _t_ground_elapsed = _time.time() - _t_ground
        log(f"  Grounding: {_t_ground_elapsed:.0f}s | {len(grounding_text)} chars")

        # Save grounding output
        with open(os.path.join(OUTPUT_DIR, "grounding_output.txt"), "a", encoding="utf-8") as gf:
            gf.write(f"=== Iter {iteration} Step {sid} ===\n{grounding_text}\n\n")

        # ---- Parse grounding result ----
        # Accept both Unicode → (U+2192, model follows prompt literally)
        # and ASCII -> (fallback). The \x{2192} doesn't work in re;
        # use the literal character in the pattern.
        labeled_pat = r'"([^"]*)"\s*(?:→|->)\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
        matches = re.findall(labeled_pat, grounding_text)

        if not matches:
            # Fallback: try bare coordinates (no label)
            bare_pat = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
            bare_matches = re.findall(bare_pat, grounding_text)
            if bare_matches:
                matches = [(f"Region {i+1}",) + bm for i, bm in enumerate(bare_matches)]
                log(f"  Used bare-coords fallback: {len(matches)} bboxes with auto-labels")

        if not matches:
            log(f"  {sid} FAILED: no coordinates found in grounding output")
            evidence[sid] = {
                'label': resolved_description,
                'value': None,
                'raw_answer': 'GROUNDING_FAILED: no coordinates found',
                'bboxes': [],
                'sub_fields': {},
            }
            states[sid] = 'FAILED'
            continue

        # ---- Validate bboxes ----
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
                    log(f"  Skipping bbox with extreme aspect ratio {aspect:.1f}: {bbox}")
                    continue

            valid_bboxes.append(bbox)
            valid_labels.append(raw_label)

        if not valid_bboxes:
            log(f"  {sid} FAILED: all bboxes failed validation")
            evidence[sid] = {
                'label': resolved_description,
                'value': None,
                'raw_answer': 'GROUNDING_FAILED: all bboxes invalid',
                'bboxes': [],
                'sub_fields': {},
            }
            states[sid] = 'FAILED'
            continue

        # Split into crop_images (first MAX_CROPS) and overflow (text-only)
        if MAX_CROPS is not None and len(valid_bboxes) > MAX_CROPS:
            cropped_bboxes = valid_bboxes[:MAX_CROPS]
            cropped_labels = valid_labels[:MAX_CROPS]
            overflow_bboxes = valid_bboxes[MAX_CROPS:]
            overflow_labels = valid_labels[MAX_CROPS:]
            overflow_count = len(overflow_bboxes)
        else:
            cropped_bboxes = valid_bboxes
            cropped_labels = valid_labels
            overflow_bboxes = []
            overflow_labels = []
            overflow_count = 0

        if overflow_count > 0:
            log(f"  {sid} Grounding: {len(valid_bboxes)} valid bboxes — "
                f"{len(cropped_bboxes)} crops, {overflow_count} text-only")
        else:
            log(f"  {sid} Grounding: {len(valid_bboxes)} valid bboxes: {cropped_labels}")

        # Draw on visualization (all bboxes, including overflow)
        for bbox, label in zip(cropped_bboxes + overflow_bboxes,
                               cropped_labels + overflow_labels):
            p_bbox = [
                int((bbox[0] / 1000.0) * orig_w),
                int((bbox[1] / 1000.0) * orig_h),
                int((bbox[2] / 1000.0) * orig_w),
                int((bbox[3] / 1000.0) * orig_h),
            ]
            overlay_color = "#00FF00" if bbox in cropped_bboxes else "#FFA500"
            draw.rectangle(p_bbox, outline=overlay_color, width=3)
            draw.text((p_bbox[0], max(0, p_bbox[1] - 12)), f"{sid}", fill=overlay_color)

        # ----------------------------------------------------------
        # Step 2b: Understanding — answer using crops + original image
        # ----------------------------------------------------------
        _t_understand = _time.time()
        total_understanding_calls += 1

        understand_content = []
        understand_content.append(build_text_content(
            "Examine the image crops and answer the question below.\n\n"
            f"=== CONTEXT FROM PREVIOUS STEPS ===\n{dep_context}\n\n"
            f"=== QUESTION ===\n{resolved_description}\n\n"
            "=== INSTRUCTIONS ===\n"
            "1. Examine each cropped region (labeled with what it contains).\n"
            "2. Use the FULL ORIGINAL IMAGE (attached last) for global context.\n"
            "3. Output values WITH POSITION INDICES so they can be aligned later.\n"
            '   Format: "[1]=value1, [2]=value2, [3]=value3, ..."\n'
            '   For structured data: "[row1_col1]=X, [row1_col2]=Y, [row2_col1]=Z, ..."\n'
            "4. Be specific with values. Include units if present.\n"
            "5. If you cannot determine a value, write [N]=? for that position.\n"
            "6. Output ONLY the indexed values. No explanation, no reasoning."
        ))

        # Attach crops with labels (up to MAX_CROPS)
        for i, (bbox, label) in enumerate(zip(cropped_bboxes, cropped_labels)):
            understand_content.append(build_text_content(
                f'--- [Region {i+1}: "{label}"] ---'
            ))
            p_bbox = [
                int((bbox[0] / 1000.0) * orig_w),
                int((bbox[1] / 1000.0) * orig_h),
                int((bbox[2] / 1000.0) * orig_w),
                int((bbox[3] / 1000.0) * orig_h),
            ]
            cropped_b64 = crop_and_encode(LOCAL_IMAGE_PATH, p_bbox)
            understand_content.append(build_image_content(cropped_b64))

        # Overflow regions: text-only (label + bbox), no image
        if overflow_count > 0:
            overflow_lines = []
            for i, (bbox, label) in enumerate(zip(overflow_bboxes, overflow_labels)):
                overflow_lines.append(
                    f"  [{len(cropped_bboxes) + i + 1}] \"{label}\" "
                    f"— bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                )
            overflow_text = (
                f"--- TEXT-ONLY REGIONS (not cropped due to limit of {MAX_CROPS}) ---\n"
                f"The following {overflow_count} regions were identified but are NOT shown "
                f"as images. Use the ORIGINAL FULL IMAGE below to locate them.\n"
                + "\n".join(overflow_lines)
            )
            understand_content.append(build_text_content(overflow_text))

        # Original image for global context
        orig_thumb_b64 = encode_pil_to_base64(
            Image.open(LOCAL_IMAGE_PATH).convert('RGB'), max_edge=1024)
        understand_content.append(build_text_content(
            "--- [FULL ORIGINAL IMAGE (for global context)] ---"
        ))
        understand_content.append(build_image_content(orig_thumb_b64))

        understand_response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=6144,
            messages=[{'role': 'user', 'content': understand_content}]
        )

        understand_text = understand_response.choices[0].message.content.strip()
        _t_understand_elapsed = _time.time() - _t_understand
        log(f"  Understanding: {_t_understand_elapsed:.0f}s | "
            f"{len(understand_text)} chars")

        # ---- Check understanding failure ----
        if not understand_text:
            evidence[sid] = {
                'label': resolved_description,
                'value': None,
                'raw_answer': 'UNDERSTANDING_FAILED: empty response',
                'bboxes': cropped_bboxes,
                'bbox_labels': cropped_labels,
                'sub_fields': {},
            }
            states[sid] = 'FAILED'
            log(f"  {sid} FAILED: understanding returned empty response")
            continue

        # Save understanding output
        with open(os.path.join(OUTPUT_DIR, "understanding_output.txt"), "a", encoding="utf-8") as uf:
            uf.write(f"=== Iter {iteration} Step {sid} ===\n{understand_text}\n\n")

        # ---- Store evidence ----
        evidence[sid] = {
            'label': resolved_description,
            'value': understand_text,   # natural language, used as-is
            'raw_answer': understand_text,
            'bboxes': cropped_bboxes,
            'bbox_labels': cropped_labels,
            'sub_fields': {},
        }
        states[sid] = 'DONE'
        log(f"  {sid} DONE: {understand_text[:120]}...")

    # ================================================================
    # Execute: compute
    # ================================================================
    elif step['type'] == '<compute>':
        # LLM compute
        total_compute_llm_calls += 1
        expression = step.get('expression', step.get('description', ''))
        compute_prompt = (
            f"Compute the following using the provided values.\n\n"
            f"=== KNOWN VALUES ===\n"
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

        comp_response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=6144,
            messages=[{'role': 'user', 'content': compute_prompt}]
        )

        comp_text = comp_response.choices[0].message.content.strip()
        result_val, _ = clean_value(extract_final_value(comp_text) or comp_text)

        evidence[sid] = {
            'label': step['description'],
            'value': result_val if result_val is not None else comp_text,
            'raw_answer': comp_text,
            'bboxes': [],
            'sub_fields': {},
        }
        states[sid] = 'DONE'
        log(f"  {sid} DONE (LLM): {step['description']} = {result_val or comp_text[:80]}")

    # ================================================================
    # Execute: logical
    # ================================================================
    elif step['type'] == '<logical>':
        # LLM logical
        total_logical_llm_calls += 1
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

        log_response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=6144,
            messages=[{'role': 'user', 'content': logical_prompt}]
        )

        log_text = log_response.choices[0].message.content.strip()

        evidence[sid] = {
            'label': step['description'],
            'value': log_text,
            'raw_answer': log_text,
            'bboxes': [],
            'sub_fields': {},
        }
        states[sid] = 'DONE'
        log(f"  {sid} DONE (LLM): {log_text[:120]}...")

    else:
        log(f"  {sid} FAILED: unknown step type '{step['type']}'")
        states[sid] = 'FAILED'

    # ---- After each step, check if done ----
    pending_after = [s for s in all_steps if states[s['id']] == 'PENDING']
    if not pending_after:
        break

_t2_elapsed = _time.time() - _t2_start

done_count = sum(1 for v in states.values() if v == 'DONE')
failed_count = sum(1 for v in states.values() if v == 'FAILED')
pending_count = sum(1 for v in states.values() if v == 'PENDING')
log(f"Phase 2 complete ({_t2_elapsed:.0f}s): {done_count} done, {failed_count} failed, "
    f"{pending_count} pending")
log(f"  API calls: {total_grounding_calls} grounding, {total_understanding_calls} understanding, "
    f"{total_compute_llm_calls} compute-LLM, {total_logical_llm_calls} logical-LLM")


# ==========================================
# 8. Phase 3: Final Reasoning
# ==========================================
_t3_start = _time.time()
log("=" * 60)
log("PHASE 3: Final Reasoning")

if FALLBACK:
    # Direct fallback: no plan, no evidence, just the original image + question
    final_prompt = (
        "Answer the following question based on the image.\n\n"
        "=== QUESTION ===\n"
        f"<q>{QUESTION}</q>\n\n"
        "=== INSTRUCTIONS ===\n"
        "1. Look at the image carefully and reason step by step.\n"
        "2. At the VERY END, enclose ONLY the final answer in <a>...</a> tags.\n"
        "3. Output the answer in the format requested by the question."
    )
else:
    # Build evidence summary (all step results as text)
    ev_summary_lines = []
    for sid in sorted(evidence.keys()):
        ev = evidence[sid]
        raw = ev.get('raw_answer') or str(ev.get('value', ''))
        if raw and raw != 'None':
            # Truncate very long evidence for final prompt
            if len(raw) > 500:
                raw = raw[:500] + "..."
            ev_summary_lines.append(f"[Step {sid} — {ev.get('label', sid)}]: {raw}")

    evidence_summary = "\n".join(ev_summary_lines) if ev_summary_lines else "(No evidence collected)"

    final_prompt = (
        "You are solving a visual reasoning problem. Below is the evidence collected "
        "from step-by-step analysis and the original image.\n\n"
        "=== COLLECTED EVIDENCE ===\n"
        f"{evidence_summary}\n\n"
        "=== ORIGINAL QUESTION ===\n"
        f"<q>{QUESTION}</q>\n\n"
        "=== INSTRUCTIONS ===\n"
        "1. The evidence contains data extracted from the image. TRUST these values.\n"
        "2. You MAY use the original image below to verify or supplement the evidence.\n"
        "3. Show your calculation steps concisely. Under 300 words.\n"
        "4. At the VERY END, enclose ONLY the final answer in <a>...</a> tags.\n"
        "5. Output the answer in the format requested by the original question."
    )

final_content = [build_text_content(final_prompt)]

# Attach original image
orig_thumb_b64 = encode_pil_to_base64(
    Image.open(LOCAL_IMAGE_PATH).convert('RGB'), max_edge=1024)
final_content.append(build_image_content(orig_thumb_b64))

final_response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.0,
    max_tokens=6144,
    messages=[{'role': 'user', 'content': final_content}]
)

final_answer = final_response.choices[0].message.content
_t3_elapsed = _time.time() - _t3_start
_t_total = _time.time() - _t1_start

log(f"Phase 3 done ({_t3_elapsed:.0f}s)")
log(f"Total pipeline: {_t_total:.0f}s")

# Extract answer from tags
final_extracted = ""
match = re.search(r'<a>(.*?)</a>', final_answer, re.DOTALL)
if match:
    final_extracted = match.group(1).strip()
else:
    lines = [l.strip() for l in final_answer.split('\n') if l.strip()]
    final_extracted = lines[-1] if lines else ""

print(f"\n{'=' * 60}")
print(f"[Final Answer]:")
print(final_answer)
print(f"\n[Extracted]: {final_extracted}")
print(f"{'=' * 60}\n")


# ==========================================
# 9. Save Results
# ==========================================

# Build serializable evidence
serializable_evidence = {}
for sid, ev in evidence.items():
    serializable_evidence[sid] = {
        'label': ev.get('label', ''),
        'value': str(ev.get('value', ''))[:200] if ev.get('value') is not None else None,
        'raw_answer': str(ev.get('raw_answer', ''))[:500],
        'bboxes': ev.get('bboxes', []),
        'bbox_labels': ev.get('bbox_labels', []),
    }

result = {
    "image_path": LOCAL_IMAGE_PATH,
    "question": QUESTION,
    "original_size": [orig_w, orig_h],
    "planning_success": planning_success,
    "plan_json": plan_json,
    "plan_text": plan_text,
    "iterations": iteration,
    "api_calls": {
        "grounding": total_grounding_calls,
        "understanding": total_understanding_calls,
        "compute_llm": total_compute_llm_calls,
        "logical_llm": total_logical_llm_calls,
        "total": (total_grounding_calls + total_understanding_calls +
                  total_compute_llm_calls + total_logical_llm_calls + 2),  # +planning +final
    },
    "step_states": states,
    "evidence": serializable_evidence,
    "final_answer": final_answer,
    "final_extracted": final_extracted,
    "timing_s": {
        "phase1_planning": round(_t1_elapsed, 1),
        "phase2_execution": round(_t2_elapsed, 1),
        "phase3_final": round(_t3_elapsed, 1),
        "total": round(_t_total, 1),
    }
}

result_path = os.path.join(OUTPUT_DIR, "pipeline_result.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
log(f"Full result saved to: {result_path}")

# Save annotated image
annotated_path = os.path.join(OUTPUT_DIR, "annotated_image.jpg")
original_img.save(annotated_path, quality=95)
log(f"Annotated image saved to: {annotated_path}")

# Save plan JSON for reference
plan_path = os.path.join(OUTPUT_DIR, "plan.json")
with open(plan_path, "w", encoding="utf-8") as f:
    json.dump(plan_json, f, ensure_ascii=False, indent=2)
log(f"Plan JSON saved to: {plan_path}")

# Append planning stats line for aggregation
stats_path = os.path.join(OUTPUT_DIR, "planning_stats.jsonl")
stats_line = json.dumps({
    "image": os.path.basename(LOCAL_IMAGE_PATH),
    "question": QUESTION[:80],
    "planning_success": planning_success,
    "n_steps": len(plan_steps),
    "fallback": FALLBACK,
    "n_evidence": len(evidence),
    "total_s": round(_t_total, 1),
}, ensure_ascii=False)
with open(stats_path, "a", encoding="utf-8") as sf:
    sf.write(stats_line + "\n")

# Print summary
log(f"Planning stats → success={planning_success}, steps={len(plan_steps)}, "
    f"evidence={len(evidence)}, fallback={FALLBACK}")

log("Pipeline complete.")
