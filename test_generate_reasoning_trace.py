import os
import json
import re
import ast
import base64
import io
import time
import pandas as pd
from PIL import Image
from openai import OpenAI


# ==========================================
# 1. 全局配置与客户端初始化
# ==========================================
# 122B for bbox grounding (strong vision)
client_122b = OpenAI(
    api_key="",
    base_url="http://10.3.112.34:22510/v1",
)

# 9B for reasoning trace parsing (text-only, no thinking mode)
client_9b = OpenAI(
    api_key="",
    base_url="http://10.3.112.34:22630/v1",
)


# ==========================================
# 2. 图像处理工具
# ==========================================
def process_base64_to_base64_str(raw_img_str, max_size=8192):
    """Decode dataset base64 image, optionally resize, re-encode."""
    img_data = base64.b64decode(raw_img_str)
    img = Image.open(io.BytesIO(img_data))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    orig_w, orig_h = img.size
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return img, (orig_w, orig_h), base64.b64encode(buffer.getvalue()).decode('utf-8')


# ==========================================
# 3. 从 GT Reasoning 解析结构化推理轨迹
# ==========================================
REASONING_PARSE_PROMPT = """/no_think
You are a reasoning trace parser. Respond with ONLY a JSON object. No thinking, no explanation, no markdown — just the JSON.

=== QUESTION ===
{question}

=== CORRECT ANSWER ===
{answer}

=== REASONING PROCESS ===
{reasoning}

=== YOUR TASK ===
Decompose the reasoning into atomic execution steps. For each step, identify its type:

- "read": reading a value from the image. description MUST say what AND where ("the number 10.1 above the 2023 bar"). Include expected_value and expected_unit.
- "compute": calculating with previously read values. Include expression.
- "logical": deduction without calculation.

CRITICAL: Each "read" targets EXACTLY ONE visual element. Split "read A and B" into separate steps.
expected_value must be copy-pasted from the reasoning. Steps numbered 1,2,3... Max 15 steps.

OUTPUT (start immediately with the brace, no preamble):
{{
  "steps": [
    {{"step_id": 1, "type": "read", "description": "...", "expected_value": "...", "expected_unit": "...", "depends_on": []}},
    {{"step_id": 2, "type": "compute", "description": "...", "expression": "...", "depends_on": [1]}}
  ],
  "final_answer": "{answer}"
}}"""


def _extract_json(text):
    """Extract JSON object from model output.

    Handles Qwen thinking-model output (thinking process + final JSON),
    markdown code blocks, and plain JSON. Uses JSONDecoder.raw_decode
    scanning from end-to-start for maximum robustness.
    """
    # 1. Remove markdown code blocks
    text = re.sub(r'```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```', '', text)

    # 2. Try direct parse of full text
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict) and 'steps' in result:
            return result
    except json.JSONDecodeError:
        pass

    # 3. Scan from end to start with raw_decode (handles arbitrary nesting)
    #    Thinking models output the final JSON at the end, so reversed scan
    #    finds it fastest and avoids partial JSON in the thinking text.
    decoder = json.JSONDecoder()
    found_dicts = []
    for i in range(len(text) - 1, -1, -1):
        if text[i] == '{':
            try:
                result, end = decoder.raw_decode(text[i:])
                if isinstance(result, dict):
                    found_dicts.append((i, result))
                    if 'steps' in result:
                        return result
            except (json.JSONDecodeError, ValueError):
                continue

    # 4. Fallback: return the last valid dict found (closest to end of text)
    if found_dicts:
        _, result = found_dicts[0]
        if isinstance(result, dict):
            return result

    print(f"  [JSON parse failed] first 300 chars: {text.strip()[:300]}...")
    return None


def parse_reasoning_to_trace(question, answer, reasoning):
    """Call LLM to parse GT reasoning into structured trace.

    Returns:
        dict with 'steps' list, or None on failure
    """
    prompt = REASONING_PARSE_PROMPT.format(
        question=question,
        answer=answer,
        reasoning=reasoning
    )

    messages = [{'role': 'user', 'content': prompt}]

    try:
        # Use 122B for more reliable JSON output (9B thinking model often
        # gets stuck overthinking and outputs template placeholders)
        response = client_122b.chat.completions.create(
            model='Qwen3.5-122B-A10B',
            temperature=0.0,
            max_tokens=12288,
            messages=messages,
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
        response_text = response.choices[0].message.content
    except Exception as e:
        print(f"  [API error]: {e}")
        return None

    return _extract_json(response_text)


# ==========================================
# 4. 为 read 步骤生成 bbox
# ==========================================
BBOX_GROUNDING_PROMPT = """/no_think
Output ONLY four numbers in brackets. No words, no explanation, no markdown.

Element to locate: {description}

RULES:
- Tight bounding box around ONLY the described element
- Coordinates normalized [0, 1000]: [xmin, ymin, xmax, ymax]
- Output format: [xmin, ymin, xmax, ymax]

Bbox:"""


def generate_bbox_for_step(description, global_base64):
    """Generate normalized bbox [xmin,ymin,xmax,ymax] for a read step.

    Returns list of 4 ints, or None on failure.
    """
    prompt = BBOX_GROUNDING_PROMPT.format(description=description)

    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url',
             'image_url': {'url': f"data:image/jpeg;base64,{global_base64}"}}
        ]
    }]

    try:
        response = client_122b.chat.completions.create(
            model='Qwen3.5-122B-A10B',
            temperature=0.0,
            max_tokens=12288,
            messages=messages,
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
        response_text = response.choices[0].message.content
    except Exception as e:
        print(f"    [bbox API error]: {e}")
        return None

    match = re.search(
        r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',
        response_text
    )
    if match:
        bbox = [int(match.group(1)), int(match.group(2)),
                int(match.group(3)), int(match.group(4))]
        if all(0 <= v <= 1000 for v in bbox) and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
            return bbox

    print(f"    [bbox parse failed]: {response_text[:100]}")
    return None


def has_meaningful_desc(desc):
    """Check if a description has semantic value."""
    if not desc or not isinstance(desc, str):
        return False
    stripped = desc.strip()
    if len(stripped) < 3:
        return False
    if not any(c.isalnum() for c in stripped):
        return False
    meaningless = {'element', 'value', 'number', 'text', 'region', 'area',
                   'box', 'item', 'label', 'data', 'point', 'figure'}
    if stripped.lower() in meaningless:
        return False
    return True


# ==========================================
# 5. 单条数据处理
# ==========================================
def process_single_row(index, row, output_dir):
    """Process one OCR-Reasoning sample: parse trace + generate bboxes + save."""
    item_id = row.get('index', index)
    question = str(row.get('question', ''))
    ground_truth = str(row.get('answer', ''))
    reasoning = str(row.get('reasoning', ''))
    raw_img_str = row.get('image', None)

    if pd.isna(raw_img_str) or not raw_img_str:
        return {"id": item_id, "status": "error", "error_msg": "missing image"}

    try:
        original_img, (orig_w, orig_h), global_base64 = \
            process_base64_to_base64_str(raw_img_str)

        # --- Stage A: Parse GT reasoning into structured steps ---
        print(f"  [Stage A] Parsing reasoning trace...")
        trace = parse_reasoning_to_trace(question, ground_truth, reasoning)

        if trace is None or 'steps' not in trace or not trace['steps']:
            return {"id": item_id, "status": "error",
                    "error_msg": "trace parsing failed"}

        steps = trace['steps']
        read_steps = [s for s in steps if s.get('type') == 'read']
        print(f"    OK: {len(steps)} steps ({len(read_steps)} read, "
              f"{len(steps) - len(read_steps)} compute/logical)")

        # --- Stage B: Generate bbox for read steps ---
        bbox_missing = sum(1 for s in read_steps
                           if 'bbox_norm' not in s or len(s.get('bbox_norm', [])) != 4)
        if bbox_missing > 0:
            print(f"  [Stage B] Generating bboxes for {bbox_missing} read steps...")
            for s in read_steps:
                if 'bbox_norm' in s and len(s.get('bbox_norm', [])) == 4:
                    continue
                desc = s.get('description', '')
                if not has_meaningful_desc(desc):
                    print(f"    Step {s['step_id']}: desc empty/meaningless, skip")
                    s['bbox_norm'] = []
                    continue
                bbox = generate_bbox_for_step(desc, global_base64)
                if bbox:
                    s['bbox_norm'] = bbox
                    print(f"    Step {s['step_id']}: {bbox}")
                else:
                    print(f"    Step {s['step_id']}: bbox generation failed")
                    s['bbox_norm'] = []

        # --- Save ---
        trace['index'] = int(item_id)
        trace['question'] = question

        safe_id = str(item_id).replace('/', '_').replace('\\', '_')
        json_path = os.path.join(output_dir, f'{safe_id}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)

        final_reads = [s for s in trace['steps'] if s.get('type') == 'read']
        bbox_ok = sum(1 for s in final_reads
                      if len(s.get('bbox_norm', [])) == 4)
        desc_ok = sum(1 for s in final_reads
                      if has_meaningful_desc(s.get('description', '')))

        return {
            "id": item_id,
            "question": question,
            "answer": ground_truth,
            "total_steps": len(trace['steps']),
            "read_steps": len(final_reads),
            "compute_steps": len(trace['steps']) - len(final_reads),
            "bbox_ok": bbox_ok,
            "desc_ok": desc_ok,
            "json_path": json_path,
            "status": "success"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"id": item_id, "status": "error", "error_msg": str(e)}


# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    tsv_path = r"C:\Users\19639\LMUData\OCR_Reasoning.tsv"
    output_dir = r"test_case/reasoning_traces"
    os.makedirs(output_dir, exist_ok=True)

    # Load target indices
    indices_path = r"test_case/random_indices.json"
    with open(indices_path, 'r', encoding='utf-8') as f:
        target_indices = json.load(f)
    print(f"Target indices: {len(target_indices)}")

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Loaded {len(df)} rows.\n")

    # Filter
    subset = df[df['index'].isin(target_indices)]
    print(f"Matched {len(subset)} target samples")

    results = []
    for i, (idx, row) in enumerate(subset.iterrows()):
        item_id = row.get('index', idx)
        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{len(subset)}] Processing ID={item_id}")
        q = str(row.get('question', ''))
        print(f"Q: {q[:150]}{'...' if len(q) > 150 else ''}")
        print(f"{'=' * 60}")

        result = process_single_row(idx, row, output_dir)
        results.append(result)

        if result['status'] == 'success':
            print(f"Result: {result['total_steps']} steps "
                  f"({result['read_steps']}r/{result['compute_steps']}c), "
                  f"bbox={result['bbox_ok']}/{result['read_steps']}, "
                  f"desc={result['desc_ok']}/{result['read_steps']}")
            print(f"Saved to: {result['json_path']}")
        else:
            print(f"Error: {result['error_msg']}")

        time.sleep(0.5)

    # Summary
    results_df = pd.DataFrame(results)
    results_output = os.path.join(output_dir, "trace_results.tsv")
    results_df.to_csv(results_output, sep='\t', index=False, encoding='utf-8')
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n{'=' * 60}")
    print(f"Done: {success_count}/{len(results)} succeeded")
    print(f"Summary: {results_output}")
    print(f"Traces directory: {output_dir}")
