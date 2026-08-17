"""
Step QA Generation Script
==========================
Reads the step_qa_pairs.xlsx (with template placeholders) and fills in proper
QA pairs using the LLM API. Supports incremental progress: saves after each
batch, can be resumed if interrupted.

Usage:
  python step_qa_generate.py                    # process all steps
  python step_qa_generate.py --start 0 --end 100  # process first 100 steps
  python step_qa_generate.py --start 500          # resume from step 500
"""

import openpyxl, re, json, os, sys, time, urllib.request
from collections import Counter
from typing import Optional

# ============================================================
# CONFIG
# ============================================================

EXCEL_PATH = "outputs/step_qa/step_qa_pairs.xlsx"
SOURCE_IMAGE_DIR = r"C:\Users\19639\LMUData\images\OCR_Reasoning"

# Load .env
def _load_env(env_path=".env"):
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, _, v = line.partition('=')
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip()
_load_env()

API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_API_BASE",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
MODEL = "qwen-plus"
BATCH_SIZE = 5  # steps per API call
SAVE_INTERVAL = 5  # save Excel every N batches

# ============================================================
# API CALLER
# ============================================================

def call_api(system: str, user: str, max_tokens: int = 4096, timeout: int = 90) -> Optional[dict]:
    """Call API and return parsed JSON, or None on failure."""
    try:
        payload = json.dumps({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }, ensure_ascii=False).encode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        url = API_BASE if API_BASE.endswith('/chat/completions') else API_BASE.rstrip('/') + '/chat/completions'
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            content = result["choices"][0]["message"]["content"].strip()
            if content.startswith("```json"): content = content[7:]
            elif content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
            return json.loads(content.strip())
    except Exception as e:
        print(f"    [API Error] {str(e)[:100]}")
        return None

# ============================================================
# PROMPTS
# ============================================================

BATCH_SYSTEM = """You are an expert at converting OCR reasoning steps into self-contained, verifiable question-answer pairs.

## Task
Given original questions and reasoning steps, convert EACH step into a QA pair.

## Rules
1. **Self-contained**: Answerable WITHOUT seeing the original image
2. **Verifiable**: Answer is a single value or an option letter (A/B/C/D)
3. **Language**: Match the step's language (Chinese → Chinese, English → English)
4. **QA type**:
   - `short_answer`: For numeric results, single values, computed answers
   - `multiple_choice`: For listing items, comparing options, picking from alternatives. Provide 4 options (A-D).
5. **Distractors**: Plausible but clearly wrong based on the step content. For enumeration steps, options should be SETS of items.
6. Use exact values from the step text — do not fabricate data.
7. **IMPORTANT**: Output valid JSON only. No trailing commas.

## Output Format
```json
{
  "qa_pairs": [
    {
      "step_index": 0,
      "qa_question": "...",
      "qa_answer": "...",
      "qa_type": "short_answer",
      "qa_options": [],
      "qa_cot": "Brief derivation"
    }
  ]
}
```"""

SINGLE_SYSTEM = """You are an expert at converting OCR reasoning steps into self-contained, verifiable question-answer pairs.
Convert the given step into a QA pair. Output JSON with fields: qa_question, qa_answer, qa_type (short_answer or multiple_choice), qa_options (array of {label, text}, empty for short_answer), qa_cot.
Make answers easily verifiable. For lists, use multiple_choice with option sets.
Match the language of the original step."""


def build_batch_prompt(steps: list[dict]) -> str:
    """Build prompt for batch QA generation."""
    parts = []
    for i, s in enumerate(steps):
        hint = ""
        if 'enumeration' in s.get('step_types', ''):
            hint = " [Hint: use multiple_choice with option sets]"
        elif 'calculation' in s.get('step_types', ''):
            hint = " [Hint: use short_answer with computed result]"
        parts.append(
            f"### Step {i}\n"
            f"Q({s['language']}): {s['question'][:200]}\n"
            f"Step {s['step_num']}/{s['total_steps']}: {s['step_text'][:400]}{hint}"
        )
    return "\n\n".join(parts)


def build_single_prompt(s: dict) -> str:
    """Build prompt for single-step QA generation."""
    hint = ""
    if 'enumeration' in s.get('step_types', ''):
        hint = "\nHint: This step lists multiple items. Consider multiple_choice with option sets as distractors."
    elif 'calculation' in s.get('step_types', ''):
        hint = "\nHint: This is a calculation step. Use short_answer with the computed value."
    return f"""Q({s['language']}): {s['question'][:200]}
Step {s['step_num']}/{s['total_steps']}: {s['step_text'][:400]}{hint}

Convert to QA pair. Output JSON only."""


# ============================================================
# GENERATION
# ============================================================

def generate_qa_batch(steps: list[dict]) -> list[Optional[dict]]:
    """Generate QA pairs for a batch of steps."""
    if len(steps) == 1:
        # Single step mode
        result = call_api(SINGLE_SYSTEM, build_single_prompt(steps[0]),
                         max_tokens=1024, timeout=60)
        return [result]

    # Batch mode
    result = call_api(BATCH_SYSTEM, build_batch_prompt(steps),
                     max_tokens=4096, timeout=120)
    if result and "qa_pairs" in result:
        return result["qa_pairs"]

    # Batch failed, fall back to individual calls
    print("    Batch failed, falling back to individual calls...")
    results = []
    for s in steps:
        qa = call_api(SINGLE_SYSTEM, build_single_prompt(s),
                     max_tokens=1024, timeout=60)
        results.append(qa)
        time.sleep(0.2)
    return results


# ============================================================
# MAIN
# ============================================================

def run_generation(start_idx: int = 0, end_idx: Optional[int] = None,
                   excel_path: str = None):
    """Fill in QA pairs from start_idx to end_idx."""
    if excel_path is None:
        excel_path = EXCEL_PATH
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    # Read all steps from Excel
    headers = [str(ws.cell(1, c).value) for c in range(1, ws.max_column + 1)]
    print(f"Reading {ws.max_row - 1} steps from {excel_path}")

    steps = []
    for row in range(2, ws.max_row + 1):
        steps.append({
            "row": row,
            "q_id": ws.cell(row, 1).value,
            "question": str(ws.cell(row, 2).value),
            "language": str(ws.cell(row, 3).value),
            "step_num": ws.cell(row, 5).value,
            "total_steps": ws.cell(row, 6).value,
            "step_text": str(ws.cell(row, 7).value),
            "step_types": str(ws.cell(row, 8).value) if ws.cell(row, 8).value else "",
            "current_gen": str(ws.cell(row, 15).value) if ws.cell(row, 15).value else "",
        })

    total = len(steps)
    if end_idx is None:
        end_idx = total

    # Filter: only process steps that need generation (template or empty)
    pending = []
    for i, s in enumerate(steps):
        if start_idx <= i < end_idx:
            gen = s["current_gen"]
            if gen in ("template", "", "template_fallback", "pending"):
                pending.append(s)

    print(f"Steps to generate: {len(pending)} (from index {start_idx} to {end_idx})")
    print(f"Skipping {total - len(pending)} already-generated steps")

    if not pending:
        print("All steps already generated!")
        return

    # Process in batches
    n_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    generated = 0
    failed = 0

    for batch_num in range(n_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(pending))
        batch = pending[batch_start:batch_end]

        print(f"Batch {batch_num+1}/{n_batches}: steps {batch_start+1}-{batch_end}...", end=" ", flush=True)

        qa_results = generate_qa_batch(batch)

        # Write results back to Excel
        for i, qa in enumerate(qa_results):
            if i >= len(batch):
                break
            s = batch[i]
            row = s["row"]

            if qa and isinstance(qa, dict) and "qa_question" in qa:
                # Convert any non-string values to string for Excel compatibility
                def _safe_str(v, default=""):
                    if v is None: return default
                    if isinstance(v, (list, dict)): return json.dumps(v, ensure_ascii=False)
                    return str(v)

                ws.cell(row, 9, _safe_str(qa.get("qa_question")))
                ws.cell(row, 10, _safe_str(qa.get("qa_answer")))
                ws.cell(row, 11, _safe_str(qa.get("qa_type", "short_answer")))
                opts_val = qa.get("qa_options", [])
                if isinstance(opts_val, list):
                    opts_val = json.dumps(opts_val, ensure_ascii=False)
                ws.cell(row, 12, opts_val)
                ws.cell(row, 13, _safe_str(qa.get("qa_cot")))
                ws.cell(row, 15, "llm_api")
                generated += 1
            else:
                ws.cell(row, 15, "api_failed")
                failed += 1

        print(f"done ({generated} ok, {failed} failed)")

        # Save periodically
        if (batch_num + 1) % SAVE_INTERVAL == 0 or batch_num == n_batches - 1:
            wb.save(excel_path)
            print(f"  [Saved to {excel_path}]")

        # Small delay between batches to avoid rate limiting
        if batch_num < n_batches - 1:
            time.sleep(1.0)

    # Final save
    wb.save(excel_path)

    print(f"\n{'='*50}")
    print(f"Generation Complete")
    print(f"  Generated: {generated}")
    print(f"  Failed: {failed}")
    print(f"  Output: {excel_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate QA pairs for step_qa_pairs.xlsx")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--excel", default=EXCEL_PATH, help="Excel file path")
    args = parser.parse_args()

    run_generation(args.start, args.end, args.excel)
