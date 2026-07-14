"""
Independent test script for OCRReasoningAgentV2 iterative execute-verify loop.
Similar to test_pgr_ocragent.py but exercises the v2 iterative architecture.

Run: python test_ocr_agent_v2.py
"""
import base64
import io
import json
import os
import re
import sys
import time as _time
from PIL import Image, ImageDraw
from openai import OpenAI


# ==========================================
# 0. Configuration — change these to test different images/questions
# ==========================================
LOCAL_IMAGE_PATH = "test_case/test_images/image14.jpg"
QUESTION = "What is the total price of the foods whose names end with the letter 'r'?"
# LOCAL_IMAGE_PATH = "test_case/test_images/image0.jpg"
# QUESTION = "What is the average number of the genres of movies made by directors who have worked in drama genres?"

# Model endpoints
PLANNER_BASE_URL = "http://10.3.112.34:22630/v1"
GROUNDER_BASE_URL = "http://10.3.112.34:22630/v1"
READER_BASE_URL = "http://10.3.112.34:22630/v1"
REASONER_BASE_URL = "http://10.3.112.34:22630/v1"

PLANNER_MODEL = "Qwen3.5-9B"
GROUNDER_MODEL = "Qwen3.5-9B"
READER_MODEL = "Qwen3.5-9B"
REASONER_MODEL = "Qwen3.5-9B"

# To use 122B for grounding, uncomment:
# GROUNDER_BASE_URL = "http://10.3.112.34:22510/v1"
# GROUNDER_MODEL = "Qwen3.5-122B-A10B"

# Crop parameters
PADDING = 60          # for ground_and_read: tight crop for OCR
SCALE_FACTOR = 3.0    # for ground_and_read: high magnification
INSPECT_PADDING = 150  # for inspect_region: generous crop for context
INSPECT_SCALE = 1.5   # for inspect_region: moderate magnification
MAX_CROP_EDGE = 1024
MAX_CROPS = 12
MAX_ITERATIONS = 20  # Sequential mode: each step = 1 iteration, plus room for re-plan

# Output
OUTPUT_DIR = "test_case/ocr_agent_v2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 1. Helper Functions
# ==========================================

def log(msg):
    """Timestamped log."""
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}")


def resize_and_encode(image_path, max_size=2048):
    """Read local image, scale, return (base64, orig_w, orig_h)."""
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


def safe_eval(expression, variables):
    """Safely evaluate a math expression."""
    import ast
    import operator as op
    safe_ops = {
        'add': op.add, 'sub': op.sub, 'mul': op.mul,
        'truediv': op.truediv, 'floordiv': op.floordiv,
        'pow': op.pow, 'mod': op.mod,
        'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum,
        'int': int, 'float': float,
    }
    allowed = set(safe_ops.keys()) | set(variables.keys())
    expr = re.sub(r'\$(\w+)', r'\1', expression)
    try:
        tree = ast.parse(expr, mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in allowed:
                raise NameError(f"'{node.id}' not allowed")
        safe_globals = {"__builtins__": {}}
        safe_globals.update(safe_ops)
        return eval(compile(tree, '<string>', 'eval'), safe_globals, variables)
    except Exception:
        return None


def clean_value(value_str):
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


def extract_final_value(text):
    """Extract the actual value from verbose model output (handles thinking traces).

    Qwen thinking models often output: <think>...</think> value (conf=0.8)
    This function strips reasoning and returns only the final value.
    """
    if not text:
        return None

    # Step 1: Extract content after last <think> / </think> / <｜end▁of▁thinking｜> tag
    for tag in ['<think>', '<｜end▁of▁thinking｜>', '<｜end▁of▁thinking｜>']:
        parts = text.split(tag)
        if len(parts) > 1:
            after = parts[-1]
            # Remove closing </think> if present
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

    # Prefer short lines (values are usually short)
    short_lines = [l for l in lines if len(l) < 200]
    if short_lines:
        text = short_lines[-1]
    else:
        text = lines[-1]

    # Step 4: Remove trailing markers like "$ (conf=0.8)"
    text = re.sub(r'\s*\(conf[^)]*\)\s*$', '', text)
    # Remove trailing " $ " patterns
    text = text.strip()

    return text

# ==========================================
# 2. Initialize Clients
# ==========================================

planner_client = OpenAI(api_key="EMPTY", base_url=PLANNER_BASE_URL)
grounder_client = OpenAI(api_key="EMPTY", base_url=GROUNDER_BASE_URL)
reader_client = OpenAI(api_key="EMPTY", base_url=READER_BASE_URL)
reasoner_client = OpenAI(api_key="EMPTY", base_url=REASONER_BASE_URL)

log(f"Planner: {PLANNER_MODEL} @ {PLANNER_BASE_URL}")
log(f"Grounder: {GROUNDER_MODEL} @ {GROUNDER_BASE_URL}")
log(f"Reader: {READER_MODEL} @ {READER_BASE_URL}")
log(f"Reasoner: {REASONER_MODEL} @ {REASONER_BASE_URL}")

# ==========================================
# 3. Preprocess Image
# ==========================================

base64_image, orig_w, orig_h = resize_and_encode(LOCAL_IMAGE_PATH, max_size=2048)
log(f"Image: {LOCAL_IMAGE_PATH} ({orig_w}x{orig_h})")
log(f"Question: {QUESTION}")


# ==========================================
# 4. Phase 1: Structured Planning
# ==========================================
_t1_start = _time.time()
log("=" * 60)
log("PHASE 1: Structured Planning (JSON Plan)")

planning_prompt = (
    f"Create a JSON execution plan to answer this question: <q>{QUESTION}</q>\n\n"
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

plan_response = planner_client.chat.completions.create(
    model=PLANNER_MODEL,
    temperature=0.0,
    max_tokens=6144,
    messages=[{
        'role': 'user',
        'content': [build_text_content(planning_prompt), build_image_content(base64_image)]
    }]
)

plan_text = plan_response.choices[0].message.content
_t1_elapsed = _time.time() - _t1_start

plan_json = parse_json_robust(plan_text)
chart_type = plan_json.get('chart_type', None) if plan_json else None
n_steps = len(plan_json.get('steps', [])) if plan_json else 0
log(f"Phase 1 done ({_t1_elapsed:.0f}s) | {n_steps} steps | chart_type={chart_type}")

if plan_json is None:
    log("FAILED to parse plan JSON! Raw output:")
    print(plan_text[:500])
    sys.exit(1)

print(f"\n{'─' * 40}\n【Plan JSON】:\n{json.dumps(plan_json, ensure_ascii=False, indent=2)}\n{'─' * 40}\n")

# Parse plan steps
plan_steps = []
raw_steps = plan_json.get('steps', [])
for i, raw in enumerate(raw_steps):
    step = {
        'id': raw.get('id', f's{i+1}'),
        'type': raw.get('type', 'ground_and_read'),
        'description': raw.get('description', ''),
        'target_desc': raw.get('target_desc', ''),
        'expected_type': raw.get('expected_type', 'text'),
        'expected_unit': raw.get('expected_unit', ''),
        'expression': raw.get('expression', ''),
        'depends_on': raw.get('depends_on', []),
        'priority': raw.get('priority', i + 1),
    }
    plan_steps.append(step)


# ==========================================
# 5. Phase 2: Iterative Execute-Verify Loop
# ==========================================
log("=" * 60)
log("PHASE 2: Iterative Execute-Verify Loop")

# Track step states
states = {s['id']: 'PENDING' for s in plan_steps}
all_steps = list(plan_steps)  # mutable copy for re-plan additions
evidence = {}  # step_id -> {label, value, confidence, bbox_norm, unit, raw}

iteration = 0
total_grounding_calls = 0
total_read_calls = 0

# Load original image once for drawing bbox annotations
original_img = Image.open(LOCAL_IMAGE_PATH).convert('RGB')
draw = ImageDraw.Draw(original_img)

while iteration < MAX_ITERATIONS:
    iteration += 1

    # ---- 1. Select ONE ready step (highest priority first) ----
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
        # Stuck — try to force unblocked steps
        ready = [s for s in pending if not s['depends_on']]
        if not ready:
            log(f"WARNING: {len(pending)} steps stuck with unmet dependencies")
            break

    ready.sort(key=lambda s: (s['priority'], s['id']))
    step = ready[0]  # Process ONE step per iteration
    sid = step['id']
    log(f"--- Iteration {iteration}/{MAX_ITERATIONS} | Step {sid} ({step['type']}) ---")

    states[sid] = 'EXECUTING'

    # ---- 2. Execute the step ----
    if step['type'] == 'ground_and_read':
        _t_ground = _time.time()
        total_grounding_calls += 1

        # --- 2a. Ground a SINGLE target ---
        grounding_prompt = (
            f"Locate this visual element in the image.\n\n"
            f"TARGET: {step['target_desc']}\n\n"
            "Copy this line EXACTLY, replacing each ? with a coordinate integer in [0, 1000]:\n"
            f"\"{sid}: {step['target_desc']}\" -> [?, ?, ?, ?]\n\n"
            "RULES:\n"
            "1. Replace ? with integers in [0, 1000] (normalized to image dimensions).\n"
            "2. Box the TIGHTEST rectangle around the target text/number.\n"
            '3. If NOT visible, replace all four ? with -1.\n'
            "4. The LAST line of your response must be the completed coordinate line. Nothing else after.\n"
            "5. The line must be character-for-character IDENTICAL except ? \u2192 numbers."
        )

        ground_response = grounder_client.chat.completions.create(
            model=GROUNDER_MODEL,
            temperature=0.0,
            max_tokens=4096,
            messages=[
                {'role': 'user', 'content': [
                    build_text_content(f"You are solving: <q>{QUESTION}</q>"),
                    build_image_content(base64_image),
                ]},
                {'role': 'assistant', 'content': plan_text[:2000]},
                {'role': 'user', 'content': [build_text_content(grounding_prompt)]},
            ]
        )

        grounding_text = ground_response.choices[0].message.content
        _t_ground_elapsed = _time.time() - _t_ground
        log(f"  Grounding: {_t_ground_elapsed:.0f}s | {len(grounding_text)} chars")

        # Save grounding output for diagnosis
        with open(os.path.join(OUTPUT_DIR, "grounding_output.txt"), "a", encoding="utf-8") as gf:
            gf.write(f"=== Iter {iteration} {sid} ===\n{grounding_text}\n\n")

        # --- 2b. Parse grounding result ---
        # Try labeled format first, then bare coordinates
        labeled_pat = r'"([^"]*)"\s*->\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
        matches = re.findall(labeled_pat, grounding_text)

        if not matches:
            # Bare coordinates: find all [x, y, x, y] patterns
            bare_pat = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
            bare_matches = re.findall(bare_pat, grounding_text)
            if bare_matches:
                bm = bare_matches[0]  # Take the first one
                label = step['target_desc'][:60]
                matches = [(f"{sid}: {label}",) + bm]
                log(f"  Parsed bare coordinates")

        if not matches:
            log(f"  {sid} FAILED: no coordinates found in grounding output")
            evidence[sid] = {
                'label': step['target_desc'], 'value': None, 'confidence': 0.0,
                'bbox_norm': None, 'unit': '', 'raw': 'NO_COORDINATES'
            }
            states[sid] = 'FAILED'
            continue

        m = matches[0]  # First match
        bbox = [int(m[1]), int(m[2]), int(m[3]), int(m[4])]

        # Parse label
        raw_label = m[0].strip()
        if ':' in raw_label:
            label = raw_label.split(':', 1)[1].strip()
        else:
            label = raw_label

        # --- 2c. Validate bbox ---
        valid = True
        if bbox[0] < 0:
            evidence[sid] = {
                'label': label, 'value': None, 'confidence': 0.0,
                'bbox_norm': bbox, 'unit': '', 'raw': f'NULL: {raw_label}'
            }
            states[sid] = 'FAILED'
            valid = False

        if valid and not (0 <= bbox[0] <= 1000 and 0 <= bbox[1] <= 1000 and
                          0 <= bbox[2] <= 1000 and 0 <= bbox[3] <= 1000):
            valid = False
        if valid and (bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
            valid = False

        area_norm = ((bbox[2] - bbox[0]) / 1000.0) * ((bbox[3] - bbox[1]) / 1000.0)

        if not valid:
            evidence[sid] = {
                'label': label, 'value': None, 'confidence': 0.1,
                'bbox_norm': bbox, 'unit': '', 'raw': f'INVALID: {raw_label}'
            }
            states[sid] = 'FAILED'
            log(f"  {sid} FAILED: bbox validation failed")
            continue

        log(f"  {sid} bbox: {bbox} (area={area_norm:.4f})")

        # --- 2d. Read a SINGLE crop ---
        _t_read = _time.time()
        total_read_calls += 1

        # Convert normalized bbox to physical pixels
        p_bbox = [
            int((bbox[0] / 1000.0) * orig_w),
            int((bbox[1] / 1000.0) * orig_h),
            int((bbox[2] / 1000.0) * orig_w),
            int((bbox[3] / 1000.0) * orig_h),
        ]
        cropped_b64 = crop_and_encode(LOCAL_IMAGE_PATH, p_bbox)

        # Draw on visualization
        draw.rectangle(p_bbox, outline="#00FF00", width=4)
        draw.text((p_bbox[0], max(0, p_bbox[1] - 15)), f"{sid}", fill="#00FF00")

        read_content = [
            build_text_content(
                "/no_think\n"
                f"Read the exact value from this cropped region.\n\n"
                f"Target: {step['target_desc']}\n"
                f"Expected type: {step['expected_type']} {step['expected_unit']}\n\n"
                "CRITICAL RULES:\n"
                "1. Output ONLY the value. NOTHING else. No reasoning, no explanation, no thinking.\n"
                "2. Your ENTIRE response must be a single line. Just the value.\n"
                "3. Examples: '2 $', '45.5 kg', 'A'\n"
                "4. If unclear: output UNCLEAR. If not found: output NOT_FOUND."
            ),
            build_image_content(cropped_b64),
        ]

        read_response = reader_client.chat.completions.create(
            model=READER_MODEL,
            temperature=0.0,
            max_tokens=512,
            messages=[{'role': 'user', 'content': read_content}]
        )

        read_text = read_response.choices[0].message.content.strip()
        _t_read_elapsed = _time.time() - _t_read
        log(f"  Reading: {_t_read_elapsed:.0f}s | raw='{read_text[:100]}'")

        # --- 2e. Parse the single reading ---
        # First extract final value from potential thinking trace
        extracted = extract_final_value(read_text)
        if extracted:
            read_text = extracted
        cleaned_val, was_cleaned = clean_value(read_text)
        if cleaned_val is None:
            conf = 0.3
        elif was_cleaned:
            conf = 0.8
        else:
            conf = 1.0

        # Extract unit from original text
        unit = ''
        if isinstance(cleaned_val, (int, float)):
            raw_no_num = re.sub(r'[\d.]+', '', read_text.replace('$', '').replace('€', ''))
            unit = raw_no_num.strip().rstrip('.,;:()')
        if not unit:
            unit = step.get('expected_unit', '')

        evidence[sid] = {
            'label': label,
            'value': cleaned_val,
            'confidence': conf,
            'bbox_norm': bbox,
            'unit': unit,
            'raw': read_text,
        }
        states[sid] = 'DONE'
        val_display = f"{cleaned_val} {unit}".strip()
        log(f"  {sid} DONE: [{label}] = {val_display} (conf={conf})")

    elif step['type'] == 'inspect_region':
        # Visual analysis — grounding + generous crop + analysis question
        _t_ground = _time.time()
        total_grounding_calls += 1

        # --- ground the target (same format as ground_and_read) ---
        grounding_prompt = (
            f"Locate this visual element in the image.\n\n"
            f"TARGET: {step['target_desc']}\n\n"
            "Copy this line EXACTLY, replacing each ? with a coordinate integer in [0, 1000]:\n"
            f"\"{sid}: {step['target_desc']}\" -> [?, ?, ?, ?]\n\n"
            "RULES:\n"
            "1. Replace ? with integers in [0, 1000] (normalized to image dimensions).\n"
            "2. Box the ENTIRE region described — not a tight corner, but the full area.\n"
            '3. If NOT visible, replace all four ? with -1.\n'
            "4. The LAST line of your response must be the completed coordinate line. Nothing else after.\n"
            "5. The line must be character-for-character IDENTICAL except ? → numbers."
        )

        ground_response = grounder_client.chat.completions.create(
            model=GROUNDER_MODEL,
            temperature=0.0,
            max_tokens=4096,
            messages=[
                {'role': 'user', 'content': [
                    build_text_content(f"You are solving: <q>{QUESTION}</q>"),
                    build_image_content(base64_image),
                ]},
                {'role': 'assistant', 'content': plan_text[:2000]},
                {'role': 'user', 'content': [build_text_content(grounding_prompt)]},
            ]
        )

        grounding_text = ground_response.choices[0].message.content
        _t_ground_elapsed = _time.time() - _t_ground
        log(f"  Grounding: {_t_ground_elapsed:.0f}s | {len(grounding_text)} chars")

        # --- parse grounding result ---
        labeled_pat = r'"([^"]*)"\s*->\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
        matches = re.findall(labeled_pat, grounding_text)

        if not matches:
            bare_pat = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
            bare_matches = re.findall(bare_pat, grounding_text)
            if bare_matches:
                bm = bare_matches[0]
                label = step['target_desc'][:60]
                matches = [(f"{sid}: {label}",) + bm]
                log(f"  Parsed bare coordinates")

        if not matches:
            log(f"  {sid} FAILED: no coordinates found in grounding output")
            evidence[sid] = {
                'label': step['target_desc'], 'value': None, 'confidence': 0.0,
                'bbox_norm': None, 'unit': '', 'raw': 'NO_COORDINATES'
            }
            states[sid] = 'FAILED'
            continue

        m = matches[0]
        bbox = [int(m[1]), int(m[2]), int(m[3]), int(m[4])]
        raw_label = m[0].strip()
        if ':' in raw_label:
            label = raw_label.split(':', 1)[1].strip()
        else:
            label = raw_label

        # --- validate bbox ---
        valid = True
        if bbox[0] < 0:
            evidence[sid] = {
                'label': label, 'value': None, 'confidence': 0.0,
                'bbox_norm': bbox, 'unit': '', 'raw': f'NULL: {raw_label}'
            }
            states[sid] = 'FAILED'
            valid = False

        if valid and not (0 <= bbox[0] <= 1000 and 0 <= bbox[1] <= 1000 and
                          0 <= bbox[2] <= 1000 and 0 <= bbox[3] <= 1000):
            valid = False
        if valid and (bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
            valid = False

        area_norm = ((bbox[2] - bbox[0]) / 1000.0) * ((bbox[3] - bbox[1]) / 1000.0)

        if not valid:
            evidence[sid] = {
                'label': label, 'value': None, 'confidence': 0.1,
                'bbox_norm': bbox, 'unit': '', 'raw': f'INVALID: {raw_label}'
            }
            states[sid] = 'FAILED'
            log(f"  {sid} FAILED: bbox validation failed")
            continue

        log(f"  {sid} bbox: {bbox} (area={area_norm:.4f})")

        # --- crop with GENEROUS padding for visual context ---
        _t_read = _time.time()
        total_read_calls += 1

        p_bbox = [
            int((bbox[0] / 1000.0) * orig_w),
            int((bbox[1] / 1000.0) * orig_h),
            int((bbox[2] / 1000.0) * orig_w),
            int((bbox[3] / 1000.0) * orig_h),
        ]
        cropped_b64 = crop_and_encode(LOCAL_IMAGE_PATH, p_bbox,
                                      padding=INSPECT_PADDING,
                                      scale_factor=INSPECT_SCALE,
                                      max_edge=MAX_CROP_EDGE)

        draw.rectangle(p_bbox, outline="#FFA500", width=4)
        draw.text((p_bbox[0], max(0, p_bbox[1] - 15)), f"{sid}", fill="#FFA500")

        # --- visual analysis question ---
        read_content = [
            build_text_content(
                "/no_think\n"
                f"Analyze this image region and answer the question.\n\n"
                f"QUESTION: {step['description']}\n"
                f"TARGET: {step['target_desc']}\n"
                f"Expected answer type: {step['expected_type']}\n\n"
                "CRITICAL RULES:\n"
                "1. Output ONLY the answer. NOTHING else. No reasoning, no explanation.\n"
                "2. Your ENTIRE response must be a single line.\n"
                "3. For counts: just the number (e.g., '3'). For booleans: YES or NO.\n"
                "4. For text values: just the text.\n"
                "5. If unclear: output UNCLEAR. If not found: output NOT_FOUND."
            ),
            build_image_content(cropped_b64),
        ]

        read_response = reader_client.chat.completions.create(
            model=READER_MODEL,
            temperature=0.0,
            max_tokens=512,
            messages=[{'role': 'user', 'content': read_content}]
        )

        read_text = read_response.choices[0].message.content.strip()
        _t_read_elapsed = _time.time() - _t_read
        log(f"  Inspect: {_t_read_elapsed:.0f}s | raw='{read_text[:120]}'")

        # --- extract value ---
        extracted = extract_final_value(read_text)
        if extracted:
            read_text = extracted
        cleaned_val, was_cleaned = clean_value(read_text)
        if cleaned_val is None:
            conf = 0.3
        elif was_cleaned:
            conf = 0.8
        else:
            conf = 1.0

        evidence[sid] = {
            'label': label,
            'value': cleaned_val,
            'confidence': conf,
            'bbox_norm': bbox,
            'unit': step.get('expected_unit', ''),
            'raw': read_text,
        }
        states[sid] = 'DONE'
        val_display = f"{cleaned_val} {step.get('expected_unit', '')}".strip()
        log(f"  {sid} DONE (inspect): [{label}] = {val_display} (conf={conf})")

    elif step['type'] == 'compute':
        # --- Execute a single compute step ---
        variables = {}
        for dep_id in step['depends_on']:
            ev = evidence.get(dep_id, {})
            if ev.get('value') is not None:
                variables[dep_id] = ev['value']
                variables[f'{dep_id}_value'] = ev['value']

        if not variables:
            states[sid] = 'FAILED'
            log(f"  {sid} FAILED: no dependencies available")
            continue

        # Clean variable values: ensure numeric
        clean_vars = {}
        for k, v in variables.items():
            if isinstance(v, str):
                cleaned, _ = clean_value(v)
                clean_vars[k] = cleaned if cleaned is not None else v
            else:
                clean_vars[k] = v

        result = safe_eval(step['expression'], clean_vars)
        if result is not None:
            evidence[sid] = {
                'label': step['description'],
                'value': result,
                'confidence': 1.0,
                'bbox_norm': None,
                'unit': '',
                'raw': f'{step["expression"]} = {result}',
            }
            states[sid] = 'DONE'
            log(f"  {sid} DONE: {step['description']} = {result}")
        else:
            # LLM fallback for complex computation
            compute_prompt = (
                f"Compute: {step['description']}\n"
                f"Expression: {step['expression']}\n"
                f"Known values:\n" +
                "\n".join(f"  {k} = {v}" for k, v in clean_vars.items()) +
                "\n\nOutput ONLY the numerical result."
            )
            comp_response = planner_client.chat.completions.create(
                model=PLANNER_MODEL, temperature=0.0, max_tokens=256,
                messages=[{'role': 'user', 'content': compute_prompt}]
            )
            try:
                val = float(comp_response.choices[0].message.content.strip())
                evidence[sid] = {
                    'label': step['description'], 'value': val,
                    'confidence': 0.9, 'bbox_norm': None,
                    'unit': '', 'raw': f'LLM: {step["expression"]} = {val}',
                }
                states[sid] = 'DONE'
                log(f"  {sid} DONE (LLM): {step['description']} = {val}")
            except ValueError:
                states[sid] = 'FAILED'
                log(f"  {sid} FAILED: LLM computation error")

    elif step['type'] == 'verify':
        # Mark verify steps as done (handled in Phase 3)
        states[sid] = 'DONE'
        log(f"  {sid} DONE (verify placeholder)")

    # ---- 3. After each step, check if all planned steps are done ----
    pending_after = [s for s in all_steps if states[s['id']] == 'PENDING']
    if not pending_after:
        # All planned steps done — check evidence quality
        low_conf = [
            s_id for s_id, ev in evidence.items()
            if ev.get('confidence', 0) < 0.5 and ev.get('bbox_norm') is not None
        ]
        null_vals = [
            s_id for s_id, ev in evidence.items()
            if ev.get('value') is None and ev.get('bbox_norm') is not None
        ]

        if (low_conf or null_vals) and iteration < MAX_ITERATIONS:
            log(f"  Evidence issues: {len(low_conf)} low-confidence, {len(null_vals)} null values")
            log("  Re-planning to fill gaps...")

            ev_lines = []
            for s_id, ev in evidence.items():
                if ev.get('value') is not None:
                    ev_lines.append(f"  [{ev['label']}] = {ev['value']} {ev.get('unit','')}")
            missing_lines = []
            for s_id in low_conf + null_vals:
                ev = evidence.get(s_id, {})
                missing_lines.append(f"  - \"{ev.get('label', s_id)}\" (confidence={ev.get('confidence', 0)})")

            replan_prompt = (
                f"Solving: <q>{QUESTION}</q>\n\n"
                "=== EVIDENCE COLLECTED ===\n" + "\n".join(ev_lines) + "\n\n"
                "=== MISSING / UNCERTAIN ===\n" + "\n".join(missing_lines) + "\n\n"
                "Generate new 'ground_and_read' steps to fill gaps. JSON array only:\n"
                '[{"id": "s_new1", "type": "ground_and_read", "description": "...", '
                '"target_desc": "...", "expected_type": "number", '
                '"expected_unit": "...", "depends_on": [], "priority": 1}]'
            )

            replan_resp = planner_client.chat.completions.create(
                model=PLANNER_MODEL, temperature=0.0, max_tokens=6144,
                messages=[{'role': 'user', 'content': replan_prompt}]
            )

            replan_json = parse_json_robust(replan_resp.choices[0].message.content)
            if replan_json:
                new_raw_steps = replan_json if isinstance(replan_json, list) else replan_json.get('steps', [])
                for j, raw in enumerate(new_raw_steps):
                    new_step = {
                        'id': raw.get('id', f's_replan{j+1}'),
                        'type': 'ground_and_read',
                        'description': raw.get('description', ''),
                        'target_desc': raw.get('target_desc', ''),
                        'expected_type': raw.get('expected_type', 'text'),
                        'expected_unit': raw.get('expected_unit', ''),
                        'expression': '',
                        'depends_on': raw.get('depends_on', []),
                        'priority': raw.get('priority', 1),
                    }
                    all_steps.append(new_step)
                    states[new_step['id']] = 'PENDING'
                log(f"  Re-plan added {len(new_raw_steps)} new steps")
            else:
                log("  Re-plan produced no parseable steps, proceeding")
                break
        else:
            break

done_count = sum(1 for v in states.values() if v == 'DONE')
failed_count = sum(1 for v in states.values() if v == 'FAILED')
pending_count = sum(1 for v in states.values() if v == 'PENDING')
log(f"Phase 2 complete: {done_count} done, {failed_count} failed, {pending_count} pending")
log(f"  {total_grounding_calls} grounding calls, {total_read_calls} read calls, {len(evidence)} evidence items")

# ==========================================
# 6. Phase 3: Final Reasoning
# ==========================================
_t3_start = _time.time()
log("=" * 60)
log("PHASE 3: Final Reasoning")

# Build evidence summary
ev_summary_lines = []
for sid, ev in evidence.items():
    if ev.get('value') is not None:
        bbox_str = ""
        if ev.get('bbox_norm'):
            b = ev['bbox_norm']
            bbox_str = f" @ [{b[0]},{b[1]},{b[2]},{b[3]}]"
        conf_str = " [LOW]" if ev.get('confidence', 1) < 0.5 else ""
        ev_summary_lines.append(
            f"[{ev['label']}] = {ev['value']} {ev.get('unit','')}{bbox_str}{conf_str}"
        )

prompt_s3 = (
    "/no_think\n"
    "You are solving a visual reasoning problem. Below is extracted evidence and the original image.\n\n"
    "=== COLLECTED EVIDENCE ===\n" +
    "\n".join(ev_summary_lines) + "\n\n"
    f"=== QUESTION ===\n<q>{QUESTION}</q>\n\n"
    "=== INSTRUCTIONS ===\n"
    "1. The evidence contains raw data extracted from the image. TRUST the values.\n"
    "2. You MAY use the original image to FILTER, CATEGORIZE, or verify evidence items.\n"
    "3. Be concise — show only the calculation steps, no extra reasoning. Under 200 words.\n"
    "4. Enclose ONLY the final answer in <a>...</a> tags.\n"
    "5. Do NOT re-read values already in evidence. Do NOT include thinking traces."
)

s3_content = [build_text_content(prompt_s3)]

# Add evidence crops (limit to max_crops)
read_ev = [(sid, ev) for sid, ev in evidence.items() if ev.get('bbox_norm') is not None]
for i, (sid, ev) in enumerate(read_ev[:MAX_CROPS]):
    s3_content.append(build_text_content(
        f'[Evidence {i+1}: "{ev["label"]}" → {ev.get("value","?")} {ev.get("unit","")}]'
    ))
    p_bbox = [
        int((ev['bbox_norm'][0] / 1000.0) * orig_w),
        int((ev['bbox_norm'][1] / 1000.0) * orig_h),
        int((ev['bbox_norm'][2] / 1000.0) * orig_w),
        int((ev['bbox_norm'][3] / 1000.0) * orig_h),
    ]
    cropped_b64 = crop_and_encode(LOCAL_IMAGE_PATH, p_bbox)
    s3_content.append(build_image_content(cropped_b64))

# Original image (scaled)
clean_img = Image.open(LOCAL_IMAGE_PATH).convert('RGB')
orig_thumb_b64 = encode_pil_to_base64(clean_img, max_edge=1024)
s3_content.append(build_image_content(orig_thumb_b64))

s3_response = reasoner_client.chat.completions.create(
    model=REASONER_MODEL,
    temperature=0.0,
    max_tokens=6144,
    messages=[{'role': 'user', 'content': s3_content}]
)

final_answer = s3_response.choices[0].message.content
_t3_elapsed = _time.time() - _t3_start
_t_total = _time.time() - _t1_start

log(f"Phase 3 done ({_t3_elapsed:.0f}s)")
log(f"Total: {_t_total:.0f}s")

# Extract answer from tags
final_extracted = ""
match = re.search(r'<a>(.*?)</a>', final_answer, re.DOTALL)
if match:
    final_extracted = match.group(1).strip()
else:
    lines = [l.strip() for l in final_answer.split('\n') if l.strip()]
    final_extracted = lines[-1] if lines else ""

print(f"\n{'=' * 60}")
print(f"【Final Answer】:")
print(final_answer)
print(f"\n【Extracted】: {final_extracted}")
print(f"{'=' * 60}\n")


# ==========================================
# 7. Save Results
# ==========================================

result = {
    "image_path": LOCAL_IMAGE_PATH,
    "question": QUESTION,
    "original_size": [orig_w, orig_h],
    "plan_json": plan_json,
    "plan_text": plan_text,
    "iterations": iteration,
    "total_grounding_calls": total_grounding_calls,
    "total_read_calls": total_read_calls,
    "step_states": states,
    "evidence": {
        sid: {
            'label': ev['label'],
            'value': str(ev.get('value', '')) if ev.get('value') is not None else None,
            'confidence': ev.get('confidence', 1.0),
            'bbox_norm': ev.get('bbox_norm'),
            'unit': ev.get('unit', ''),
        }
        for sid, ev in evidence.items()
    },
    "final_answer": final_answer,
    "final_extracted": final_extracted,
    "timing_s": {
        "phase1": round(_t1_elapsed, 1),
        "phase3": round(_t3_elapsed, 1),
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

log("Pipeline complete.")
