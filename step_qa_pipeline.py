"""
Step-level QA Pair Construction Pipeline
=========================================
Parses OCR Reasoning steps and converts each step into a verifiable QA pair.
Supports LLM-based generation (via OpenAI-compatible API) and template-based fallback.

Input:  OCR_Reasoning.xlsx (200 questions with reasoning ground truth)
Output: step_qa_pairs.xlsx (one row per QA pair)
        step_qa_output/images/ (copied images for verification)
"""

import openpyxl
import re
import json
import os
import sys
import shutil
import time
from pathlib import Path
from collections import Counter
from typing import Optional

# ============================================================
# CONFIG
# ============================================================

INPUT_EXCEL = "outputs/Qwen3.5-9B-OCRAgentV5/T20260730_G533ef742/Qwen3.5-9B-OCRAgentV5_OCR_Reasoning.xlsx"
OUTPUT_EXCEL = "outputs/step_qa/step_qa_pairs.xlsx"
OUTPUT_IMAGE_DIR = "outputs/step_qa/images"
SOURCE_IMAGE_DIR = r"C:\Users\19639\LMUData\images\OCR_Reasoning"

# Load .env file for API credentials
def _load_env(env_path: str = ".env"):
    """Load environment variables from .env file."""
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip()
                    if key not in os.environ:  # don't override existing env vars
                        os.environ[key] = value

_load_env()

# LLM API config (OpenAI-compatible)
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_API_BASE",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
MODEL_NAME = "qwen-plus"
LLM_BATCH_SIZE = 5  # steps per API batch call
MAX_WORKERS = 3  # parallel API workers (keep low to avoid rate limits)

# ============================================================
# STEP PARSER
# ============================================================

def parse_cn_steps(text: str) -> list[str]:
    """Parse Chinese reasoning into steps. Returns steps WITHOUT final answer statement."""
    parts = re.split(r'(?:第[一二三四五六七八九十百零\d]+步)[：:]', text)
    if not parts:
        return []
    # Remove the "最终的回答是: ..." part from the last step
    if '最终的回答是' in parts[-1]:
        parts[-1] = parts[-1].split('最终的回答是')[0].strip()
    # Also try to extract the final answer for reference
    steps = [s.strip() for s in parts if s.strip()]
    return steps


def parse_en_steps(text: str) -> list[str]:
    """Parse English reasoning into steps. Returns steps WITHOUT final answer statement."""
    parts = re.split(r'(?:step\s*\d+)\s*:?\s*', text, flags=re.IGNORECASE)
    if not parts:
        return []
    if 'the final answer is' in parts[-1].lower():
        parts[-1] = re.split(r'the final answer is', parts[-1], flags=re.IGNORECASE)[0].strip()
    steps = [s.strip() for s in parts if s.strip()]
    return steps


def classify_step(step: str) -> list[str]:
    """Classify step operation types for prompt customization."""
    types = []
    if re.search(r'(识别|找出|找到|查阅|查看|观察|读取|提取|从.*中|确认|确定.*为|根据.*知|由图|图中|表格|图片|海报|菜单|Identify|identify|find|locate|read|extract|observe|check|look at|from the|in the image|in the poster|in the chart|in the table|in the menu|shown in|according to)', step, re.IGNORECASE):
        types.append('extraction')
    if re.search(r'[+\-*/×÷=＝%]|计算|求和|相加|相减|相乘|除以|总和|合计|平均|差值|乘积|乘以|差|和|比率|占|百分|公式|calculate|compute|sum|total|average|difference|multiply|divide|add|subtract|product|ratio|percent|formula|count', step, re.IGNORECASE):
        types.append('calculation')
    if re.search(r'(比较|对比|相比|大于|小于|高于|低于|超过|最大|最小|最高|最低|最多|最少|compare|greater|less|higher|lower|more than|less than|maximum|minimum)', step, re.IGNORECASE):
        types.append('comparison')
    if re.search(r'(分别|包括|包含|列举|例如|respectively|including|list|following|such as|namely)', step, re.IGNORECASE):
        types.append('enumeration')
    if re.search(r'(因为|所以|因此|由于|判断|推理|得出|结论|总结|综合|综上|because|therefore|thus|hence|conclude|judge|reason|decide|infer|conclusion)', step, re.IGNORECASE):
        types.append('reasoning')
    if not types:
        types.append('other')
    return types


# ============================================================
# QA PROMPT BUILDER
# ============================================================

SYSTEM_PROMPT = """You are an expert at converting OCR reasoning steps into self-contained, verifiable question-answer pairs.

## Task
Given an original question and one reasoning step, create a QA pair that:
1. Is self-contained (does NOT require seeing the original image)
2. Has an answer that can be automatically verified (exact match for short answers, option letter for multiple choice)
3. Preserves the reasoning skill being tested in that step

## Output Format
Return ONLY a JSON object:
```json
{
  "qa_question": "The question text (in the same language as the original)",
  "qa_answer": "The correct answer",
  "qa_type": "multiple_choice" or "short_answer",
  "qa_options": [{"label": "A", "text": "option text"}, ...],
  "qa_cot": "Brief explanation of how to derive the answer from the step"
}
```

## Rules
- Language: Match the original question's language (Chinese → Chinese QA, English → English QA)
- For steps with LONG lists (>5 items): Use multiple_choice with options that each contain a set of items
- For steps with a single numeric/text result: Use short_answer
- For steps with LaTeX formulas: Express as multiple_choice with formula options
- For steps comparing values: Can use short_answer (just the result) or multiple_choice
- Do NOT include the final answer from later steps — each QA is independent
- The distractors in multiple_choice must be plausible but clearly wrong
- Keep questions concise, max 2 sentences
"""


def build_qa_prompt(question: str, step_text: str, step_num: int, total_steps: int,
                    step_types: list[str], language: str) -> str:
    """Build a prompt for LLM to generate a QA pair from a reasoning step."""
    type_hint = ""
    if 'enumeration' in step_types and 'extraction' in step_types:
        type_hint = "\n**Hint**: This step involves identifying/extracting multiple items. Consider using multiple_choice with option sets."
    elif 'calculation' in step_types:
        type_hint = "\n**Hint**: This step involves calculation. Consider using short_answer with the computed result."
    elif 'comparison' in step_types:
        type_hint = "\n**Hint**: This step involves comparison. Multiple_choice or short_answer both work."
    elif 'reasoning' in step_types:
        type_hint = "\n**Hint**: This step involves logical reasoning/judgment. Consider multiple_choice if there are distinct alternatives."

    return f"""Original question ({language}): {question}

Step {step_num} of {total_steps}:
{step_text}{type_hint}

Convert this step into a verifiable QA pair. Output JSON only."""


# ============================================================
# LLM API CALLER
# ============================================================

def call_llm_api(system_prompt: str, user_prompt: str,
                 api_key: str = "", api_base: str = "",
                 model: str = "qwen-plus") -> Optional[dict]:
    """Call OpenAI-compatible API (DashScope) to generate QA pair. Returns parsed JSON or None."""
    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        }, ensure_ascii=False).encode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # API_BASE may already include the full path; handle both cases
        url = api_base
        if not url.endswith('/chat/completions'):
            url = url.rstrip('/') + '/chat/completions'

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            content = result["choices"][0]["message"]["content"]
            # Extract JSON from response (handle markdown code blocks)
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            return json.loads(content)
    except Exception as e:
        print(f"  [API Error] {e}")
        return None


def call_llm_api_long(system_prompt: str, user_prompt: str,
                      api_key: str = "", api_base: str = "",
                      model: str = "qwen-plus") -> Optional[dict]:
    """Call API with longer max_tokens for batch processing."""
    try:
        import urllib.request
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
        }, ensure_ascii=False).encode('utf-8')
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        url = api_base
        if not url.endswith('/chat/completions'):
            url = url.rstrip('/') + '/chat/completions'
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            content = result["choices"][0]["message"]["content"].strip()
            if content.startswith("```json"): content = content[7:]
            elif content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
            return json.loads(content.strip())
    except Exception as e:
        print(f"  [API Error] {e}")
        return None


def call_llm_batch(prompts: list[dict], api_key: str, api_base: str,
                   model: str = "qwen-plus") -> list[Optional[dict]]:
    """Call LLM API for multiple prompts. Each prompt is {system, user, step_key}."""
    results = []
    for p in prompts:
        qa = call_llm_api(p["system"], p["user"], api_key, api_base, model)
        if qa is not None:
            qa["_step_key"] = p.get("step_key", "")
        results.append(qa)
        if len(prompts) > 1:
            time.sleep(0.3)  # rate limiting
    return results


BATCH_QA_SYSTEM_PROMPT = """You are an expert at converting OCR reasoning steps into self-contained, verifiable question-answer pairs.

## Task
Given an original question and a list of reasoning steps (from different problems), convert EACH step into a QA pair.

## Rules for each QA pair
1. **Self-contained**: The QA should be answerable WITHOUT seeing the original image. Include all necessary context.
2. **Verifiable**: Answer should be automatically gradable (exact match for short answers, option letter for multiple choice).
3. **Language**: Match the step's original language.
4. **QA type selection**:
   - `short_answer`: For steps with a single numeric/text result (calculation, single value extraction). Keep answer concise.
   - `multiple_choice`: For steps listing multiple items, comparing options, or where distractors are natural. Provide 4 options (A-D) with sets or values.
5. **Distractors**: Wrong options must be PLAUSIBLE but clearly incorrect based on the step content.
6. **Step text as context**: Use the exact values/items from the step text in your QA — do NOT fabricate new data.

## Output Format
Return a JSON object with a "qa_pairs" array:
```json
{
  "qa_pairs": [
    {
      "step_index": 0,
      "qa_question": "...",
      "qa_answer": "...",
      "qa_type": "short_answer",
      "qa_options": [],
      "qa_cot": "Brief derivation from step to answer"
    },
    ...
  ]
}
```"""


def build_batch_qa_prompt(steps_batch: list[dict]) -> str:
    """Build a prompt for batch QA generation from multiple steps."""
    parts = []
    for i, step_info in enumerate(steps_batch):
        type_hint = ""
        if 'enumeration' in step_info.get('step_types', []):
            type_hint = " [Hint: consider multiple_choice with option sets]"
        elif 'calculation' in step_info.get('step_types', []):
            type_hint = " [Hint: consider short_answer with computed result]"

        parts.append(
            f"### Step {i}\n"
            f"Original question ({step_info.get('language', '')}): {step_info['question'][:200]}\n"
            f"Step {step_info['step_num']}/{step_info['total_steps']}: {step_info['step_text'][:400]}{type_hint}"
        )
    return "\n\n".join(parts)


def generate_qa_batch(steps_batch: list[dict], api_key: str, api_base: str,
                      model: str = "qwen-plus") -> list[Optional[dict]]:
    """Generate QA pairs for a batch of steps in a single API call."""
    user_prompt = build_batch_qa_prompt(steps_batch)
    result = call_llm_api_long(BATCH_QA_SYSTEM_PROMPT, user_prompt, api_key, api_base, model)
    if result is None or "qa_pairs" not in result:
        # Fall back to individual calls
        print("    [falling back to individual calls]")
        results = []
        for s in steps_batch:
            prompt = build_qa_prompt(
                s['question'], s['step_text'], s['step_num'], s['total_steps'],
                s.get('step_types', []), s.get('language', '')
            )
            qa = call_llm_api(SYSTEM_PROMPT, prompt, api_key, api_base, model)
            if qa is not None:
                qa["_gen_method"] = "llm_api_individual"
            results.append(qa)
        return results
    # Tag each result with generation method
    for qa in result["qa_pairs"]:
        if isinstance(qa, dict):
            qa["_gen_method"] = "llm_api_batch"
    return result["qa_pairs"]


# ============================================================
# TEMPLATE-BASED QA GENERATOR (fallback when API unavailable)
# ============================================================

def generate_qa_from_template(step_text: str, step_types: list[str],
                               language: str) -> dict:
    """Generate a basic QA pair using rule-based templates.
    This is a fallback — results should be reviewed manually."""
    step_clean = step_text.strip()

    # Try to extract a clear answer from the step
    # Pattern 1: "= number" or "is number" or "结果是 number"
    result_match = re.search(
        r'(?:[=＝]|is|are|结果|答案为|得出|得到|共计|总和为|总共|总|为)\s*([0-9,.]+(?:\s*[%％]|\s*元|\s*克|\s*万|\s*亿|\s*十亿|\s*美元|\s*天|\s*个|\s*份|\s*人|\s*次|\s*种|\s*项|\s*站)?)',
        step_clean)
    if result_match:
        answer = result_match.group(1).strip()
        # Build a question from the step's first sentence or action description
        action_match = re.search(r'(?:第一步|第二步|第三步|第四步|第五步|第六步|第七步|第八步|step\d+)?:?\s*(.+)', step_clean)
        q_prefix = action_match.group(1)[:80] if action_match else step_clean[:80]
        return {
            "qa_question": f"根据以上信息，{q_prefix}？" if 'cn' in _lang_simplify(language) else f"Based on the above, what is the result of: {q_prefix}?",
            "qa_answer": answer,
            "qa_type": "short_answer",
            "qa_options": [],
            "qa_cot": step_clean[:200]
        }

    # Pattern 2: Multiple choice from labeled items
    items = re.findall(r'(?:[A-E][.、)]|\([A-E]\))\s*(.+?)(?=(?:[A-E][.、)]|\([A-E]\)|$))', step_clean)
    if len(items) >= 2:
        correct_idx = 0
        # Try to find which one is indicated as correct
        correct_match = re.search(r'(正确|对|correct|right|true|✓|✅|√)', step_clean)
        options = []
        for i, item in enumerate(items[:5]):
            label = chr(65 + i)  # A, B, C, D, E
            options.append({"label": label, "text": item.strip()[:100]})
        return {
            "qa_question": "根据以上信息，哪个选项是正确的？" if 'cn' in _lang_simplify(language) else "Based on the above, which option is correct?",
            "qa_answer": "A" if not correct_match else "B",
            "qa_type": "multiple_choice",
            "qa_options": options,
            "qa_cot": step_clean[:200]
        }

    # Pattern 3: Generic short answer
    return {
        "qa_question": f"请回答：{step_clean[:100]}..." if 'cn' in _lang_simplify(language) else f"Answer: {step_clean[:100]}...",
        "qa_answer": "[需要人工标注]",
        "qa_type": "short_answer",
        "qa_options": [],
        "qa_cot": step_clean[:200]
    }


def _lang_simplify(lang: str) -> str:
    return 'cn' if 'chinese' in lang.lower() else 'en'


# ============================================================
# MAIN PIPELINE
# ============================================================

class StepQAPipeline:
    def __init__(self, input_excel: str, output_excel: str,
                 output_image_dir: str, source_image_dir: str,
                 api_key: str = "", api_base: str = "", model: str = "qwen-plus"):
        self.input_excel = input_excel
        self.output_excel = output_excel
        self.output_image_dir = output_image_dir
        self.source_image_dir = source_image_dir
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.use_api = bool(api_key)

    def run(self, start_row: int = 2, end_row: Optional[int] = None,
            max_steps: Optional[int] = None):
        """Execute the full pipeline with batch LLM processing."""
        wb = openpyxl.load_workbook(self.input_excel)
        ws = wb.active

        if end_row is None:
            end_row = ws.max_row

        # Create image output directory
        os.makedirs(self.output_image_dir, exist_ok=True)

        # ================================================================
        # PHASE 1: Parse all steps and collect metadata
        # ================================================================
        print("Phase 1: Parsing all steps...")
        all_step_records = []  # Will hold all step data for Excel output

        total_questions = 0
        skipped_last_steps = 0

        for row in range(start_row, end_row + 1):
            q_id = int(ws.cell(row, 1).value)
            question = str(ws.cell(row, 2).value)
            reasoning = str(ws.cell(row, 4).value)
            language = str(ws.cell(row, 9).value)

            # Parse steps
            if 'chinese' in language.lower():
                steps = parse_cn_steps(reasoning)
            else:
                steps = parse_en_steps(reasoning)

            if not steps:
                continue

            # Skip last step if it's just an answer statement
            last_step = steps[-1]
            is_answer_statement = bool(
                re.search(r'(最终的回答是|the final answer is|故.*正确|故.*错误|综上.*选|因此.*选|The answer is|答案是|所以选|正确选项|D正确|C正确|B正确|A正确|只有.*符合)',
                          last_step, re.IGNORECASE)
            )
            steps_to_convert = steps[:-1] if is_answer_statement else steps
            if is_answer_statement:
                skipped_last_steps += 1

            # Copy image
            img_src = os.path.join(self.source_image_dir, f"{q_id}.jpg")
            img_dst = os.path.join(self.output_image_dir, f"{q_id}.jpg")
            img_rel_path = f"images/{q_id}.jpg"
            try:
                if os.path.exists(img_src):
                    shutil.copy2(img_src, img_dst)
                else:
                    img_rel_path = f"[missing: {img_src}]"
            except Exception as e:
                img_rel_path = f"[error: {e}]"

            total_questions += 1

            for step_idx, step_text in enumerate(steps_to_convert):
                step_num = step_idx + 1
                step_types = classify_step(step_text)

                all_step_records.append({
                    "q_id": q_id,
                    "question": question,
                    "language": language,
                    "reasoning": reasoning,
                    "step_num": step_num,
                    "total_steps": len(steps_to_convert),
                    "step_text": step_text,
                    "step_types": step_types,
                    "img_rel_path": img_rel_path,
                    # QA fields to be filled
                    "qa_question": "",
                    "qa_answer": "",
                    "qa_type": "",
                    "qa_options": [],
                    "qa_cot": "",
                    "gen_method": ""
                })

        print(f"  Parsed {len(all_step_records)} steps from {total_questions} questions")
        print(f"  Skipped {skipped_last_steps} final answer-statement steps")

        # ================================================================
        # PHASE 2: Generate QA pairs using LLM (parallel batch mode)
        # ================================================================
        if self.use_api and all_step_records:
            total_steps = len(all_step_records)
            if max_steps:
                total_steps = min(total_steps, max_steps)

            print(f"\nPhase 2: Generating QA pairs via {self.model} ({total_steps} steps, "
                  f"batch={LLM_BATCH_SIZE}, workers={MAX_WORKERS})...")

            # Prepare all batches
            batches = []
            for batch_start in range(0, total_steps, LLM_BATCH_SIZE):
                batch_end = min(batch_start + LLM_BATCH_SIZE, total_steps)
                batch = all_step_records[batch_start:batch_end]
                batch_input = []
                for rec in batch:
                    batch_input.append({
                        "question": rec["question"],
                        "step_text": rec["step_text"],
                        "step_num": rec["step_num"],
                        "total_steps": rec["total_steps"],
                        "step_types": rec["step_types"],
                        "language": rec["language"],
                    })
                batches.append((batch_start, batch, batch_input))

            print(f"  Total batches: {len(batches)}")

            # Process batches in parallel
            from concurrent.futures import ThreadPoolExecutor, as_completed

            completed = 0
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_batch = {}
                for batch_start, batch, batch_input in batches:
                    future = executor.submit(
                        generate_qa_batch, batch_input,
                        self.api_key, self.api_base, self.model
                    )
                    future_to_batch[future] = (batch, batch_input)

                for future in as_completed(future_to_batch):
                    batch, batch_input = future_to_batch[future]
                    try:
                        qa_results = future.result(timeout=120)
                    except Exception as e:
                        print(f"  Batch failed: {e}")
                        qa_results = [None] * len(batch)

                    # Fill QA results back
                    for i, qa in enumerate(qa_results):
                        if i < len(batch):
                            if qa and isinstance(qa, dict) and "qa_question" in qa:
                                batch[i]["qa_question"] = qa.get("qa_question", "")
                                batch[i]["qa_answer"] = qa.get("qa_answer", "")
                                batch[i]["qa_type"] = qa.get("qa_type", "short_answer")
                                batch[i]["qa_options"] = qa.get("qa_options", [])
                                batch[i]["qa_cot"] = qa.get("qa_cot", "")
                                batch[i]["gen_method"] = qa.get("_gen_method", "llm_api")
                            else:
                                qa_t = generate_qa_from_template(
                                    batch[i]["step_text"], batch[i]["step_types"],
                                    batch[i]["language"]
                                )
                                for k, v in qa_t.items():
                                    batch[i][k] = v
                                batch[i]["gen_method"] = "template_fallback"

                    completed += 1
                    if completed % 10 == 0 or completed == len(batches):
                        print(f"  Progress: {completed}/{len(batches)} batches done "
                              f"(~{completed * LLM_BATCH_SIZE} steps)")

            # Trim to max_steps if specified
            if max_steps:
                all_step_records = all_step_records[:max_steps]
        else:
            # No API: use template for all
            print("\nPhase 2: Generating QA pairs using templates (no API)...")
            for rec in all_step_records[:max_steps]:
                qa = generate_qa_from_template(rec["step_text"], rec["step_types"], rec["language"])
                rec["qa_question"] = qa.get("qa_question", "")
                rec["qa_answer"] = qa.get("qa_answer", "")
                rec["qa_type"] = qa.get("qa_type", "short_answer")
                rec["qa_options"] = qa.get("qa_options", [])
                rec["qa_cot"] = qa.get("qa_cot", "")
                rec["gen_method"] = "template"

        # ================================================================
        # PHASE 3: Write to Excel
        # ================================================================
        print(f"\nPhase 3: Writing output Excel...")
        out_wb = openpyxl.Workbook()
        out_ws = out_wb.active
        out_ws.title = "Step QA Pairs"
        out_headers = [
            "question_id", "question", "language", "full_reasoning",
            "step_number", "total_steps", "step_text", "step_types",
            "qa_question", "qa_answer", "qa_type", "qa_options",
            "qa_cot", "image_path", "generation_method"
        ]
        for col, header in enumerate(out_headers, 1):
            out_ws.cell(1, col, header)

        out_row = 2
        for rec in all_step_records:
            row_data = [
                rec["q_id"], rec["question"], rec["language"], rec["reasoning"],
                rec["step_num"], rec["total_steps"], rec["step_text"],
                ",".join(rec["step_types"]),
                rec["qa_question"], rec["qa_answer"], rec["qa_type"],
                json.dumps(rec.get("qa_options", []), ensure_ascii=False),
                rec["qa_cot"], rec["img_rel_path"], rec["gen_method"]
            ]
            for col, value in enumerate(row_data, 1):
                out_ws.cell(out_row, col, value)
            out_row += 1

        out_wb.save(self.output_excel)

        # ================================================================
        # Summary
        # ================================================================
        print(f"\n{'='*60}")
        print(f"Pipeline Complete")
        print(f"{'='*60}")
        print(f"Questions processed: {total_questions}")
        print(f"QA pairs generated: {len(all_step_records)}")
        print(f"Last steps skipped: {skipped_last_steps}")
        gen_counts = Counter(r["gen_method"] for r in all_step_records)
        for method, count in gen_counts.most_common():
            print(f"  {method}: {count}")
        print(f"Output Excel: {self.output_excel}")
        print(f"Output Images: {self.output_image_dir} ({len(os.listdir(self.output_image_dir))} files)")
        print(f"{'='*60}")

        return len(all_step_records)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build step-level QA pairs from OCR Reasoning pipeline results")
    parser.add_argument("--input", default=INPUT_EXCEL,
                        help="Input Excel file path")
    parser.add_argument("--output", default=OUTPUT_EXCEL,
                        help="Output Excel file path")
    parser.add_argument("--image-dir", default=OUTPUT_IMAGE_DIR,
                        help="Output directory for copied images")
    parser.add_argument("--source-images", default=SOURCE_IMAGE_DIR,
                        help="Source directory for original images")
    parser.add_argument("--api-key", default=API_KEY,
                        help="OpenAI-compatible API key")
    parser.add_argument("--api-base", default=API_BASE,
                        help="API base URL")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="Model name for API calls")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps to process (for testing)")
    parser.add_argument("--start-row", type=int, default=2,
                        help="Starting row in input Excel")
    parser.add_argument("--end-row", type=int, default=None,
                        help="Ending row in input Excel")

    args = parser.parse_args()

    pipeline = StepQAPipeline(
        input_excel=args.input,
        output_excel=args.output,
        output_image_dir=args.image_dir,
        source_image_dir=args.source_images,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model
    )

    pipeline.run(
        start_row=args.start_row,
        end_row=args.end_row,
        max_steps=args.max_steps
    )
