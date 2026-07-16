"""
将 reasoning ground truth 转换为 V4 Planning JSON Schema 格式的伪标签。

输入: test_case/reasoning_traces/*.json (reasoning GT)
输出: test_case/planning_pseudo_labels/*.json (V4 planning format)

转换规则:
  - "read" → "ground-understand"
  - step_id (int) → id="step_N" (str)
  - depends_on [1,2] → ["step_1","step_2"]
  - description: LLM 根据 GT 信息生成详细的可执行描述 (不含答案值)

用法:
    python convert_gt_to_planning.py                          # 全部
    python convert_gt_to_planning.py --indices 100,101,13     # 指定索引
    python convert_gt_to_planning.py --limit 10               # 前 N 个
    python convert_gt_to_planning.py --dry-run                # 只看不调 LLM
"""
import os
import sys
import json
import argparse
import time

API_BASE = "http://10.3.112.34:22630/v1"
API_KEY = "sk-EMPTY"
MODEL_NAME = "Qwen3.5-9B"
MAX_TOKENS = 4096
INPUT_DIR = "test_case/reasoning_traces"
OUTPUT_DIR = "test_case/planning_pseudo_labels"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Prompt
# ============================================================================

CONVERSION_SYSTEM = """You are a planning data annotator. You will be given a question and a reasoning ground-truth with step-by-step reasoning. Your job is to convert the reasoning steps into a clean execution plan in JSON format.

## Conversion Rules
1. Map step types: "read" → "ground-understand", "compute" remains "compute", "logical" remains "logical".
2. Rename IDs: integer step_id → string "step_N" (e.g., 1 → "step_1").
3. Convert depends_on: [1, 2] → ["step_1", "step_2"].
4. **Rewrite descriptions**: the GT descriptions are terse (e.g., "reading the height value 152.6 from the image"). You MUST expand them into detailed, self-contained executor instructions that:
   - Mention WHAT visual element to locate (using spatial hints from bbox positions where helpful)
   - State WHAT type of information to extract (number, text, label, etc.)
   - Do NOT include the expected answer value — the executor must read it from the image
5. The description should be specific enough that an executor can find the right region without seeing the GT values.

## CRITICAL — What NOT to include
- Do NOT include expected_value or unit in descriptions (the executor doesn't know them)
- Do NOT include the final answer in any step
- Do NOT include bbox coordinates in descriptions (use natural language spatial hints instead)
- Keep descriptions under 200 characters

## Example
Input reasoning GT:
{
  "question": "预算100元，最多可以点几份汤？",
  "steps": [
    {"step_id": 1, "type": "read", "description": "读取菜单中纯牛肉丸汤的价格", "depends_on": []},
    {"step_id": 2, "type": "compute", "description": "计算预算除以单价", "depends_on": [1]}
  ]
}

Your output:
{
  "steps": [
    {"id": "step_1", "type": "ground-understand", "description": "Locate the menu item '纯牛肉丸汤' in the image and read its price", "depends_on": []},
    {"id": "step_2", "type": "compute", "description": "Divide the budget (100) by the unit price from step_1 to find how many servings can be bought", "depends_on": ["step_1"]}
  ]
}

## Output Format
Output ONLY the JSON object. NO other text, no markdown fences, no thinking."""

CONVERSION_USER = """Convert this reasoning GT into a V4 planning JSON.

Question: <q>{question}</q>

Reasoning GT (steps with expected values — DO NOT include these values in your output):
```json
{gt_steps_json}
```

Output ONLY the JSON plan:"""


# ============================================================================
# LLM Client
# ============================================================================

def get_openai_client():
    from openai import OpenAI
    return OpenAI(base_url=API_BASE, api_key=API_KEY)


def convert_one(client, gt_data, dry_run=False):
    """Convert a single reasoning trace to V4 planning format."""
    question = gt_data.get('question', '')
    steps = gt_data.get('steps', [])
    index = gt_data.get('index', 'unknown')

    # Strip expected_value and bbox from the input (keep description, type, depends_on)
    gt_steps_stripped = []
    for s in steps:
        gt_steps_stripped.append({
            'step_id': s.get('step_id'),
            'type': s.get('type'),
            'description': s.get('description', ''),
            'depends_on': s.get('depends_on', []),
        })

    user_prompt = CONVERSION_USER.format(
        question=question,
        gt_steps_json=json.dumps(gt_steps_stripped, ensure_ascii=False, indent=2),
    )

    if dry_run:
        print(f"\n[{index}] DRY-RUN — would send:")
        print(f"  Question: {question[:80]}")
        print(f"  Steps: {len(steps)}")
        print(f"  User prompt length: {len(user_prompt)} chars")
        return None

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
    # Try fence extraction
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try outermost brace
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert reasoning GT to V4 planning pseudo-labels')
    parser.add_argument('--indices', type=str, default=None,
                        help='Comma-separated indices to process (e.g. "100,101,13")')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without calling LLM')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect input files
    all_files = sorted(
        [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')],
        key=lambda x: int(x.replace('.json', ''))
    )

    if args.indices:
        target = set(args.indices.split(','))
        all_files = [f for f in all_files if f.replace('.json', '') in target]

    if args.limit:
        all_files = all_files[:args.limit]

    print(f"Processing {len(all_files)} files...")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {args.output_dir}")
    print(f"Mode:   {'DRY-RUN' if args.dry_run else 'LIVE (LLM)'}")
    print()

    if not args.dry_run:
        client = get_openai_client()
    else:
        client = None

    success = 0
    failed = 0
    skipped = 0
    t_start = time.time()

    for fname in all_files:
        idx = fname.replace('.json', '')
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(args.output_dir, fname)

        # Skip if already exists
        if os.path.exists(out_path) and not args.dry_run:
            print(f"[{idx}] SKIP (already exists)")
            skipped += 1
            continue

        with open(in_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        t0 = time.time()
        result = convert_one(client, gt_data, dry_run=args.dry_run)
        elapsed = time.time() - t0

        if args.dry_run:
            continue

        if result is None:
            print(f"[{idx}] FAILED — parsing error")
            failed += 1
            continue

        plan, raw = result
        if plan is None or 'steps' not in plan:
            print(f"[{idx}] FAILED — invalid plan: {str(raw)[:100]}")
            failed += 1
            continue

        # Validate and enrich output
        output = {
            'index': gt_data.get('index', idx),
            'question': gt_data.get('question', ''),
            'plan': plan,
            'gt_steps_count': len(gt_data.get('steps', [])),
            'gt_final_answer': gt_data.get('final_answer', ''),
        }

        # Validate step IDs match GT
        gt_step_count = len(gt_data.get('steps', []))
        plan_step_count = len(plan['steps'])
        if gt_step_count != plan_step_count:
            print(f"[{idx}] WARNING: step count mismatch — GT={gt_step_count}, plan={plan_step_count}")

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"[{idx}] OK ({elapsed:.1f}s) — {plan_step_count} steps")
        success += 1

    total = time.time() - t_start
    print(f"\nDone. success={success} failed={failed} skipped={skipped} total_time={total:.0f}s")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
