"""
将 Excel 中缺少 reasoning_trace 的 9 道题目的 reasoning GT（自由文本）
转换为 V4 Planning JSON Schema 格式。

与 convert_gt_to_planning.py 不同：输入是自由文本 reasoning，不是结构化 steps。

用法:
    python convert_missing_to_planning.py
"""
import os
import sys
import json
import time
import pandas as pd

API_BASE = "http://10.3.112.34:22630/v1"
API_KEY = "sk-EMPTY"
MODEL_NAME = "Qwen3.5-9B"
MAX_TOKENS = 4096

EXCEL_PATH = "outputs/Qwen3.5-9B-OCRAgentV4/Qwen3.5-9B-OCRAgentV4_OCR_Reasoning_qwen-plus.xlsx"
OUTPUT_DIR = "test_case/planning_pseudo_labels"

# 9 indices without reasoning traces
MISSING_INDICES = [218, 529, 594, 769, 789, 851, 994, 1001, 1060]

os.makedirs(OUTPUT_DIR, exist_ok=True)

CONVERSION_SYSTEM = """You are a planning data annotator. You will be given a question and its reasoning ground-truth (free text). Your job is to:
1. Parse the reasoning into logical steps
2. Convert those steps into a clean execution plan in JSON format

## Step Type Mapping
- "ground-understand": Locate and read a specific piece of information from the image (text, number, label, color, shape, etc.)
- "compute": Perform a calculation using results from previous steps. No image access needed.
- "logical": Make a judgment, comparison, or filter using results from previous steps. No image access needed.

## Conversion Rules
1. Each atomic operation in the reasoning becomes one step.
2. Steps that read from the image → "ground-understand"
3. Steps that perform calculations → "compute"
4. Steps that compare/judge/filter → "logical"
5. depends_on must reference earlier step IDs (only steps that this step genuinely needs as input).
6. depends_on should be [] for steps that don't need prior results.

## Description Requirements
- Rewrite descriptions into detailed, self-contained executor instructions.
- Mention WHAT visual element to locate (use spatial hints like "top-left", "in the legend", "next to the bar", etc.)
- State WHAT type of information to extract (number, text, label, color, etc.)
- Do NOT include the expected answer value — the executor must read it from the image.
- Keep under 200 characters per description.

## CRITICAL — What NOT to include
- Do NOT include expected values or the final answer in any step.
- Do NOT include bbox coordinates.
- Descriptions must be PURE instructions, not conclusions.

## Example

Input:
  Question: "预算100元，最多可以点几份汤？"
  Reasoning: "第一步：在菜单中找到'纯牛肉丸汤'，其价格为20元。第二步：用预算100除以单价20，得到5份。"

Output:
{
  "steps": [
    {"id": "step_1", "type": "ground-understand", "description": "Locate the menu item '纯牛肉丸汤' in the image and read its unit price", "depends_on": []},
    {"id": "step_2", "type": "compute", "description": "Divide the budget (100) by the unit price from step_1 to get the maximum number of servings", "depends_on": ["step_1"]}
  ]
}

## Output Format
Output ONLY the JSON object. NO other text, no markdown fences, no thinking."""

CONVERSION_USER = """Convert this reasoning GT into a V4 planning JSON.

Question: <q>{question}</q>

Reasoning GT (free text — DO NOT include these values in your plan descriptions):
{reasoning}

Output ONLY the JSON plan:"""


def get_openai_client():
    from openai import OpenAI
    return OpenAI(base_url=API_BASE, api_key=API_KEY)


def convert_free_text(client, question, reasoning, idx):
    """Convert free-text reasoning to V4 planning format."""
    user_prompt = CONVERSION_USER.format(
        question=question,
        reasoning=reasoning,
    )

    messages = [
        {'role': 'system', 'content': CONVERSION_SYSTEM},
        {'role': 'user', 'content': user_prompt},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        extra_body={
            'guided_json': {
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
        }
    )

    raw = response.choices[0].message.content
    plan = _parse_json(raw)
    return plan, raw


def _parse_json(text):
    import re
    if text is None:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def main():
    df = pd.read_excel(EXCEL_PATH)

    # Build lookup: index -> row
    idx_map = {}
    for _, row in df.iterrows():
        idx_map[row['index']] = row

    print(f"Processing {len(MISSING_INDICES)} missing indices (no reasoning trace)...")
    print(f"Input:  {EXCEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    client = get_openai_client()
    success = 0
    failed = 0
    skipped = 0
    t_start = time.time()

    for idx in MISSING_INDICES:
        out_path = os.path.join(OUTPUT_DIR, f"{idx}.json")

        if os.path.exists(out_path):
            print(f"[{idx}] SKIP (already exists)")
            skipped += 1
            continue

        if idx not in idx_map:
            print(f"[{idx}] MISSING — not in Excel")
            failed += 1
            continue

        row = idx_map[idx]
        question = str(row['question'])
        reasoning = str(row.get('reasoning', ''))

        if not reasoning or reasoning == 'nan':
            print(f"[{idx}] FAILED — no reasoning in Excel")
            failed += 1
            continue

        print(f"[{idx}] Q: {question[:60]}...")
        print(f"       Reasoning: {len(reasoning)} chars")

        t0 = time.time()
        result = convert_free_text(client, question, reasoning, idx)
        elapsed = time.time() - t0

        if result is None:
            print(f"[{idx}] FAILED — parsing error ({elapsed:.1f}s)")
            failed += 1
            continue

        plan, raw = result
        if plan is None or 'steps' not in plan:
            print(f"[{idx}] FAILED — invalid plan: {str(raw)[:100]} ({elapsed:.1f}s)")
            failed += 1
            continue

        # Validate depends_on references
        all_ids = {s['id'] for s in plan['steps']}
        valid = True
        for s in plan['steps']:
            for dep in s.get('depends_on', []):
                if dep not in all_ids:
                    print(f"[{idx}] WARNING — broken dep: {s['id']} -> {dep}")
                    valid = False
        if not valid:
            print(f"[{idx}] FAILED — broken dependencies ({elapsed:.1f}s)")
            failed += 1
            continue

        output = {
            'index': idx,
            'question': question,
            'plan': plan,
            'gt_steps_count': len(plan['steps']),
            'gt_final_answer': str(row.get('answer', '')),
            'source': 'excel_reasoning_column',
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"[{idx}] OK ({elapsed:.1f}s) — {len(plan['steps'])} steps")
        success += 1

    total = time.time() - t_start
    print(f"\nDone. success={success} failed={failed} skipped={skipped} total_time={total:.0f}s")


if __name__ == '__main__':
    main()
