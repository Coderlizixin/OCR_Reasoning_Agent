"""
Step Reasoning QA Evaluation (GPT-Judge Version)
=================================================
Evaluates Qwen3.5-9B-API on the Step Reasoning QA benchmark.
Uses the SAME approach as VLMEvalKit's OCR_Reasoning evaluation:
  - Same prompt structure (image + question + format instruction)
  - GPT-4o-mini judge for answer extraction from verbose model predictions
  - Matches the OCR_Reasoning build_prompt format exactly
"""
import openpyxl
import json
import os
import sys
import re
import time
import base64
import math
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd


# ============================================================
# CONFIG — matches Qwen3.5-9B-API in vlmeval/config.py EXACTLY
# ============================================================

MODEL_NAME = "Qwen3.5-9B"
API_BASE = "http://10.3.112.34:22630/v1/chat/completions"
API_KEY = "sk-EMPTY"
TEMPERATURE = 0.0
MAX_TOKENS = 6144
IMG_DETAIL = "high"
API_NPROC = 4
RETRY = 3

# Extra body for image processing — matches Qwen3.5-9B-API in vlmeval/config.py
EXTRA_BODY = {
    "mm_processor_kwargs": {
        "images_kwargs": {
            "size": {
                "longest_edge": 8192 * 32 * 32,
                "shortest_edge": 4096 * 32 * 32,
            }
        }
    }
}

# GPT Judge config — uses qwen-plus (DashScope), same API base as model
# Note: OCR_Reasoning via run.py uses gpt-4o-mini (OpenAI).
# Our environment uses qwen-plus as an equivalent GPT judge.
JUDGE_MODEL = "qwen-plus"
# Load judge API key from .env
def _load_env(path=".env"):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, _, v = line.partition('=')
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip()
_load_env()

JUDGE_API_KEY = os.environ.get("OPENAI_API_KEY", "")
JUDGE_API_BASE = os.environ.get("OPENAI_API_BASE",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

# Input/output
STEP_QA_EXCEL = "outputs/step_qa/step_qa_pairs.xlsx"
IMAGE_DIR = "outputs/step_qa/images"
FALLBACK_IMAGE_DIR = r"C:\Users\19639\LMUData\images\OCR_Reasoning"
OUTPUT_DIR = f"outputs/Step_Reasoning_QA/{MODEL_NAME}"
OUTPUT_EXCEL = f"{OUTPUT_DIR}/{MODEL_NAME}_Step_Reasoning_QA.xlsx"
OUTPUT_DETAILS = f"{OUTPUT_DIR}/{MODEL_NAME}_Step_Reasoning_QA_step_details.xlsx"
OUTPUT_SUMMARY = f"{OUTPUT_DIR}/{MODEL_NAME}_Step_Reasoning_QA_step_summary.csv"


# ============================================================
# API CALLER (same as Qwen3.5-9B-API)
# ============================================================

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_api(messages: list[dict], model: str = MODEL_NAME,
             api_base: str = API_BASE, api_key: str = API_KEY,
             max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE,
             retry: int = RETRY) -> Optional[str]:
    """Call OpenAI-compatible API."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, ensure_ascii=False).encode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    for attempt in range(retry):
        try:
            req = urllib.request.Request(api_base, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retry - 1:
                time.sleep((attempt + 1) * 5)
            else:
                return None


# ============================================================
# PROMPT BUILDER — matches OCR_Reasoning.build_prompt EXACTLY
# ============================================================
# Original OCR_Reasoning.build_prompt:
#   msgs = [image, question + instruction]
#   instruction (CN): "\n通过一步一步的推理解决这个复杂的问题。最后的回答的组成应该是: '{format}'."
#   instruction (EN): "\nSolve the complex problem through step-by-step reasoning. The composition of the final answer should be: '{format}'."

def build_prompt_messages(question_text: str, image_path: str) -> list[dict]:
    """Build messages with image + text. Same structure as OCR_Reasoning."""
    b64 = encode_image(image_path)
    data_url = f"data:image/jpeg;base64,{b64}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url, "detail": IMG_DETAIL}},
                {"type": "text", "text": question_text},
            ]
        }
    ]


# ============================================================
# GPT JUDGE — same approach as OcrR_auxeval in ocr_reasoning.py
# ============================================================

JUDGE_EXAMPLES_CN = """
示例1:
问题: 2023年品牌二的收入是多少（单位：十亿元人民币）？
模型回答: 2023年总收入为10.1十亿元人民币，品牌二占比为7%。因此，2023年品牌二的收入 = 10.1 * 7% = 10.1 * 0.07 = 0.707 十亿元人民币。
提取答案: 0.707

示例2:
问题: 图中用哪种颜色的柱子表示'世界对中国经济的依存度'？
模型回答: 根据图例，黑色柱子对应"世界对中国经济的依存度"。经过仔细观察，2023年柱状图中黑色部分占比较小。最终答案是黑色。
提取答案: 黑色

示例3:
问题: 2000年、2007年、2012年、2017年这四年中，数值依次是哪些？
选项: A. [0.4, 0.7, 1.1, 1.2] B. [0.4, 0.7, 1.0, 1.2] C. [0.5, 0.7, 1.1, 1.2] D. [0.4, 0.6, 1.1, 1.2]
模型回答: 2000年对应0.4，2007年对应0.7，2012年对应1.1，2017年对应1.2。比较选项，这些数值与选项A完全匹配。所以正确答案是A。
提取答案: A
"""

JUDGE_EXAMPLES_EN = """
Example 1:
Question: How many days does this campaign last?
Model response: The campaign runs from June 3 to June 22. 22 - 3 = 19, plus 1 since both days are included. So the answer is 20 days.
Extracted answer: 20

Example 2:
Question: What is the next holiday in the poster?
Options: A. Mother's Day B. Christmas C. New Year D. Thanksgiving
Model response: The poster date is October 31st. Mother's Day is in May (past), Christmas is December 25 (upcoming). So the answer is B (Christmas).
Extracted answer: B

Example 3:
Question: How many times does the word 'HOTEL' appear in the poster?
Model response: Looking at the poster: PALACE HOTEL, HOTEL D'ORANCE, HOTEL KURHAUS, GRAND HOTEL, SAVOY HOTEL, HOTEL RAUCH. That's 6 occurrences.
Extracted answer: 6
"""


def build_judge_prompt_cn(original_question: str, step_questions: list[dict],
                          prediction: str) -> str:
    """Build GPT judge prompt to extract step answers from prediction (Chinese)."""
    parts = [
        "请从以下AI助手的回答中，提取每个子问题的最终答案。\n",
        "以下是几个提取答案的示例：\n",
        JUDGE_EXAMPLES_CN,
        "\n---\n",
        f"原问题: {original_question}\n",
        "子问题列表:\n",
    ]
    for s in step_questions:
        sq = s["qa_question"]
        opts = s.get("qa_options", "")
        if opts and opts != "[]":
            try:
                options = json.loads(opts)
                opt_str = " 选项: " + " ".join(f"{o['label']}. {o['text']}" for o in options)
                parts.append(f"  子问题{s['step_num']}: {sq}{opt_str}\n")
            except:
                parts.append(f"  子问题{s['step_num']}: {sq}\n")
        else:
            parts.append(f"  子问题{s['step_num']}: {sq}\n")

    parts.append(f"\nAI助手的回答:\n{prediction}\n")
    parts.append(
        "\n请按以下格式输出每个子问题的提取答案（只输出答案，不要解释）：\n"
        "子问题1: [提取的答案]\n"
        "子问题2: [提取的答案]\n"
        "注意：选择题只输出选项字母（如A），简答题输出最终数值或结果。\n"
    )
    return "".join(parts)


def build_judge_prompt_en(original_question: str, step_questions: list[dict],
                          prediction: str) -> str:
    """Build GPT judge prompt to extract step answers from prediction (English)."""
    parts = [
        "Extract the final answer for each sub-question from the AI assistant's response below.\n\n",
        "Here are some examples of answer extraction:\n\n",
        JUDGE_EXAMPLES_EN,
        "\n---\n\n",
        f"Original question: {original_question}\n\n",
        "Sub-questions:\n",
    ]
    for s in step_questions:
        sq = s["qa_question"]
        opts = s.get("qa_options", "")
        if opts and opts != "[]":
            try:
                options = json.loads(opts)
                opt_str = " Options: " + " ".join(f"{o['label']}. {o['text']}" for o in options)
                parts.append(f"  Sub-question {s['step_num']}: {sq}{opt_str}\n")
            except:
                parts.append(f"  Sub-question {s['step_num']}: {sq}\n")
        else:
            parts.append(f"  Sub-question {s['step_num']}: {sq}\n")

    parts.append(f"\nAI assistant's response:\n{prediction}\n\n")
    parts.append(
        "Output the extracted answer for each sub-question in the following format "
        "(output ONLY the answer, no explanation):\n"
        "Sub-question 1: [extracted answer]\n"
        "Sub-question 2: [extracted answer]\n"
        "Note: For multiple-choice, output only the option letter (e.g., A). "
        "For short-answer, output the final value/result.\n"
    )
    return "".join(parts)


def parse_judge_response(judge_response: str, step_count: int,
                         language: str = "English") -> dict:
    """Parse GPT judge's extracted answers."""
    results = {}
    is_cn = "chinese" in language.lower()

    for sn in range(1, step_count + 1):
        if is_cn:
            pat = rf'(?:子问题|步骤)\s*{sn}\s*[：:]\s*(.+?)(?=(?:子问题|步骤)\s*{sn+1}\s*[：:]|\Z)'
        else:
            pat = rf'(?:Sub-question|[Ss]tep)\s*{sn}\s*[：:]\s*(.+?)(?=(?:Sub-question|[Ss]tep)\s*{sn+1}\s*[：:]|\Z)'

        match = re.search(pat, judge_response, re.DOTALL)
        if match:
            raw = match.group(1).strip()
            # Take first meaningful line
            lines = [l.strip() for l in raw.split('\n') if l.strip()]
            raw = lines[0] if lines else raw
            raw = re.sub(r'\*+', '', raw)
            raw = raw.strip('.,;:;!?*[]()"\'` \t\n\r')
            results[f"step{sn}"] = raw if raw else ""
        else:
            results[f"step{sn}"] = ""

    return results


# ============================================================
# ANSWER CHECKING (same logic as post_check in ocr_reasoning.py)
# ============================================================

def check_step_answer(predicted: str, ground_truth: str,
                      qa_type: str = "short_answer",
                      qa_options: str = "") -> bool:
    """Robust answer comparison with format leniency.

    Rules:
    1. Multiple choice: accept letter (A/B/C/D) OR full option content
    2. Numeric: use relative tolerance (5%), no strict decimal requirement
    3. Format: strip units, punctuation, whitespace before comparing
    """
    if not predicted:
        return False

    pred = predicted.strip()
    gt = ground_truth.strip()

    # ── Normalize: strip common wrapper chars ──
    pred_clean = pred.strip('.,;:;!?*[]()"\'` \t\n\r')
    gt_clean = gt.strip('.,;:;!?*[]()"\'` \t\n\r')

    # ── Exact match (case-insensitive) ──
    if pred_clean.lower() == gt_clean.lower():
        return True

    # ── Multiple choice: flexible matching ──
    if qa_type == "multiple_choice":
        # Rule 1a: model outputs option letter (A/B/C/D)
        choice_match = re.search(r'\b([A-Da-d])\b', pred_clean)
        if choice_match:
            letter = choice_match.group(1).upper()
            if letter == gt_clean.upper().strip():
                return True
            # Rule 1b: model output letter, GT is the option content → check if
            # the letter's option text matches GT
            if qa_options and qa_options != "[]":
                try:
                    options = json.loads(qa_options)
                    for opt in options:
                        if isinstance(opt, str) and opt.strip().startswith(letter + "."):
                            opt_text = opt.strip()[2:].strip()  # remove "A."
                            if opt_text.lower() == gt_clean.lower():
                                return True
                        elif isinstance(opt, dict) and opt.get("label", "").upper() == letter:
                            if opt.get("text", "").lower() == gt_clean.lower():
                                return True
                except:
                    pass

        # Rule 1c: model outputs full content, GT is a letter → find matching option
        if qa_options and qa_options != "[]":
            try:
                options = json.loads(qa_options)
                for opt in options:
                    opt_text = ""
                    opt_label = ""
                    if isinstance(opt, str):
                        opt_label = opt.strip()[:1].upper()
                        opt_text = opt.strip()[2:].strip()
                    elif isinstance(opt, dict):
                        opt_label = opt.get("label", "").upper()
                        opt_text = opt.get("text", "")
                    # Check if pred matches option content
                    if opt_text.lower() == pred_clean.lower():
                        if opt_label == gt_clean.upper().strip():
                            return True
                        # Also accept if GT is the content itself
                        if opt_text.lower() == gt_clean.lower():
                            return True
                    # Partial overlap for long option texts
                    if len(opt_text) > 10 and len(pred_clean) > 5:
                        pred_words = set(pred_clean.lower().replace('、', ' ').replace(',', ' ').split())
                        opt_words = set(opt_text.lower().replace('、', ' ').replace(',', ' ').split())
                        if pred_words and opt_words:
                            overlap = len(pred_words & opt_words) / max(len(pred_words), len(opt_words))
                            if overlap > 0.7 and (opt_label == gt_clean.upper().strip() or opt_text.lower() == gt_clean.lower()):
                                return True
            except:
                pass

    # ── Numeric comparison with relaxed tolerance ──
    units_cn = ['%', '元', '十亿元', '十亿', '万', '亿', '千', '百', '克', '天', '个', '份', '人', '次', '种', '项', '站',
                '人民币', '美元', '米', '厘米', '毫米', '公里', '小时', '分钟', '秒', '辆', '张', '盒', '瓶', '串', '斤', '两']
    units_en = ['%', '$', '¥', 'dollars', 'yuan', 'days', 'hours', 'minutes', 'grams', 'kg', 'km', 'm', 'cm', 'mm']

    pred_num = re.sub(r'\s+', '', pred_clean)
    gt_num = re.sub(r'\s+', '', gt_clean)
    for u in units_cn + units_en + [',']:
        pred_num = pred_num.replace(u, '')
        gt_num = gt_num.replace(u, '')

    try:
        pv = float(pred_num)
        gv = float(gt_num)
        # Relaxed tolerance: 1% relative or 0.01 absolute
        if math.isclose(pv, gv, rel_tol=0.01, abs_tol=0.01):
            return True
        # Also check: significant digits match (e.g., 0.7 ≈ 0.707)
        if gv != 0 and abs(pv - gv) / abs(gv) < 0.02:
            return True
    except (ValueError, TypeError):
        pass

    # ── Unit-stripped string match ──
    pred_stripped = pred_clean
    gt_stripped = gt_clean
    for u in units_cn + units_en:
        pred_stripped = pred_stripped.replace(u, '')
        gt_stripped = gt_stripped.replace(u, '')
    pred_stripped = re.sub(r'\s+', '', pred_stripped).lower()
    gt_stripped = re.sub(r'\s+', '', gt_stripped).lower()
    if pred_stripped == gt_stripped:
        return True

    # ── Normalized string (remove all whitespace) ──
    pred_norm = re.sub(r'\s+', '', pred_clean).lower()
    gt_norm = re.sub(r'\s+', '', gt_clean).lower()
    if pred_norm == gt_norm:
        return True

    # ── Substring match for long text answers ──
    if len(gt_clean) > 20 and len(pred_clean) > 5:
        gt_words = set(gt_clean.lower().replace('、', ' ').replace(',', ' ').replace('，', ' ').split())
        pred_words_set = set(pred_clean.lower().replace('、', ' ').replace(',', ' ').replace('，', ' ').split())
        if gt_words and pred_words_set:
            overlap = len(gt_words & pred_words_set) / len(gt_words)
            if overlap > 0.6:
                return True

    return False


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(start_idx: int = 0, end_idx: Optional[int] = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================
    # Phase 1: Build evaluation records from step_qa_pairs.xlsx
    # ================================================================
    print("Phase 1: Building question prompts (matching OCR_Reasoning style)...")
    wb = openpyxl.load_workbook(STEP_QA_EXCEL)
    ws = wb.active

    questions = defaultdict(list)
    for row in range(2, ws.max_row + 1):
        q_id = int(ws.cell(row, 1).value)
        questions[q_id].append({
            "step_num": ws.cell(row, 5).value,
            "total_steps": ws.cell(row, 6).value,
            "step_text": str(ws.cell(row, 7).value),
            "qa_question": str(ws.cell(row, 9).value),
            "qa_answer": str(ws.cell(row, 10).value),
            "qa_type": str(ws.cell(row, 11).value),
            "qa_options": str(ws.cell(row, 12).value),
            "qa_cot": str(ws.cell(row, 13).value),
        })

    # Get original question info
    wb_in = openpyxl.load_workbook(
        "outputs/Qwen3.5-9B-OCRAgentV5/T20260730_G533ef742/Qwen3.5-9B-OCRAgentV5_OCR_Reasoning.xlsx"
    )
    ws_in = wb_in.active
    orig_info = {}
    for row in range(2, ws_in.max_row + 1):
        q_id = int(ws_in.cell(row, 1).value)
        orig_info[q_id] = {
            "orig_question": str(ws_in.cell(row, 2).value),
            "language": str(ws_in.cell(row, 9).value),
            "task": str(ws_in.cell(row, 7).value) if ws_in.cell(row, 7).value else "",
        }

    # Build prompts matching OCR_Reasoning style:
    # Original: "通过一步一步的推理解决这个复杂的问题。最后的回答的组成应该是: '{format}'."
    # We use: "通过一步一步的推理解决以下问题。\n[step questions]\n按顺序回答每个子问题。"
    eval_records = []
    for q_id in sorted(questions.keys()):
        steps = questions[q_id]
        total = steps[0]["total_steps"]
        info = orig_info.get(q_id, {})
        lang = info.get("language", "English")
        is_cn = "Chinese" in lang

        # Build question_text in OCR_Reasoning style
        if is_cn:
            parts = ["通过一步一步的推理解决以下问题。\n"]
            for s in steps:
                sn = s["step_num"]
                sq = s["qa_question"]
                st = s["qa_type"]
                opts = s["qa_options"]
                if st == "multiple_choice" and opts and opts != "[]":
                    try:
                        options = json.loads(opts)
                        opt_text = " ".join(f"{o['label']}. {o['text']}" for o in options)
                        parts.append(f"问题{sn}: {sq}\n选项: {opt_text}")
                    except:
                        parts.append(f"问题{sn}: {sq}")
                else:
                    parts.append(f"问题{sn}: {sq}")
            parts.append(
                "\n按顺序输出每个问题的答案。答案格式："
                "\n问题1: [答案]\n问题2: [答案]\n...\n"
                "注意：选择题只输出选项字母（A/B/C/D），简答题直接输出最终数值或结果。"
            )
        else:
            parts = ["Solve the following problems through step-by-step reasoning.\n"]
            for s in steps:
                sn = s["step_num"]
                sq = s["qa_question"]
                st = s["qa_type"]
                opts = s["qa_options"]
                if st == "multiple_choice" and opts and opts != "[]":
                    try:
                        options = json.loads(opts)
                        opt_text = " ".join(f"{o['label']}. {o['text']}" for o in options)
                        parts.append(f"Question {sn}: {sq}\nOptions: {opt_text}")
                    except:
                        parts.append(f"Question {sn}: {sq}")
                else:
                    parts.append(f"Question {sn}: {sq}")
            parts.append(
                "\nOutput your answer for each question in order. Answer format:"
                "\nQuestion 1: [answer]\nQuestion 2: [answer]\n...\n"
                "Note: For multiple-choice, output only the option letter (A/B/C/D). "
                "For short-answer, output the final value or result directly."
            )

        prompt_text = "\n".join(parts)

        # Build GT answers
        gt_answers = {}
        for s in steps:
            gt_answers[f"step{s['step_num']}"] = {
                "answer": s["qa_answer"],
                "type": s["qa_type"],
            }

        # Find image
        img_path = os.path.join(IMAGE_DIR, f"{q_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(FALLBACK_IMAGE_DIR, f"{q_id}.jpg")
        if not os.path.exists(img_path):
            continue

        eval_records.append({
            "question_id": q_id,
            "prompt": prompt_text,
            "image_path": img_path,
            "language": lang,
            "task": info.get("task", ""),
            "step_count": total,
            "gt_answers": gt_answers,
            "steps": steps,
            "orig_question": info.get("orig_question", ""),
        })

    total_questions = len(eval_records)
    print(f"  Built {total_questions} evaluation records")

    if end_idx is not None:
        eval_records = eval_records[start_idx:end_idx]
    elif start_idx > 0:
        eval_records = eval_records[start_idx:]

    # ================================================================
    # Phase 2: Model inference
    # ================================================================
    n_eval = len(eval_records)
    print(f"\nPhase 2: Running inference on {n_eval} questions ({MODEL_NAME})...")

    def infer_one(idx: int, record: dict) -> dict:
        msgs = build_prompt_messages(record["prompt"], record["image_path"])
        response = call_api(msgs)
        return {
            "index": idx,
            "question_id": record["question_id"],
            "language": record["language"],
            "task": record["task"],
            "step_count": record["step_count"],
            "prompt": record["prompt"],
            "prediction": response if response else "[API_ERROR]",
            "gt_answers": json.dumps(record["gt_answers"], ensure_ascii=False),
            "steps_json": json.dumps(record["steps"], ensure_ascii=False),
            "orig_question": record["orig_question"],
        }

    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=API_NPROC) as executor:
        futures = {executor.submit(infer_one, i, rec): i
                   for i, rec in enumerate(eval_records)}
        for future in as_completed(futures):
            try:
                result = future.result(timeout=300)
                results.append(result)
                completed += 1
                if completed % 20 == 0 or completed == n_eval:
                    print(f"  Progress: {completed}/{n_eval}", flush=True)
            except Exception as e:
                i = futures[future]
                results.append({
                    "index": i,
                    "question_id": eval_records[i]["question_id"],
                    "prediction": f"[ERROR: {e}]",
                })
                completed += 1

    results.sort(key=lambda x: x["index"])

    # ================================================================
    # Phase 3: GPT Judge — extract step answers from predictions
    # ================================================================
    print(f"\nPhase 3: GPT Judge ({JUDGE_MODEL}) extracting step answers...")

    detailed_rows = []
    total_steps = 0
    correct_steps = 0
    correct_questions = 0
    judge_completed = 0

    for r in results:
        q_id = r["question_id"]
        lang = r["language"]
        task = r["task"]
        step_count = r["step_count"]
        prediction = r["prediction"]
        orig_question = r.get("orig_question", "")
        steps = json.loads(r.get("steps_json", "[]"))
        gt_answers = json.loads(r.get("gt_answers", "{}"))

        # Build GPT judge prompt
        is_cn = "Chinese" in lang
        if is_cn:
            judge_prompt = build_judge_prompt_cn(orig_question, steps, prediction)
        else:
            judge_prompt = build_judge_prompt_en(orig_question, steps, prediction)

        # Call GPT judge
        judge_messages = [{"role": "user", "content": judge_prompt}]
        judge_response = call_api(
            judge_messages,
            model="qwen-plus",  # Use qwen-plus as judge (available on DashScope)
            api_base=JUDGE_API_BASE,
            api_key=JUDGE_API_KEY,
            max_tokens=1024,
            temperature=0.0,
        )

        # Parse judge response
        if judge_response:
            pred_answers = parse_judge_response(judge_response, step_count, lang)
        else:
            pred_answers = {f"step{sn}": "" for sn in range(1, step_count + 1)}

        # Check each step
        all_correct = True
        step_details = {}
        for sn in range(1, step_count + 1):
            sk = f"step{sn}"
            gt_step = gt_answers.get(sk, {})
            gt_answer = gt_step.get("answer", "") if isinstance(gt_step, dict) else str(gt_step)
            qa_type = gt_step.get("type", "short_answer") if isinstance(gt_step, dict) else "short_answer"
            pred_answer = pred_answers.get(sk, "")

            step_opts = ""
            for si in steps:
                if si.get('step_num') == sn:
                    step_opts = si.get('qa_options', '')
                    break
            is_correct = check_step_answer(pred_answer, gt_answer, qa_type, step_opts)
            total_steps += 1
            if is_correct:
                correct_steps += 1
            else:
                all_correct = False

            step_details[sk] = {
                "predicted": pred_answer,
                "ground_truth": gt_answer,
                "correct": is_correct,
                "type": qa_type,
            }

        if all_correct and step_count > 0:
            correct_questions += 1

        # Store judge_response back into r for flat table generation
        r["_judge_response"] = judge_response if judge_response else ""

        detailed_rows.append({
            "question_id": q_id,
            "language": lang,
            "task": task,
            "step_count": step_count,
            "all_correct": all_correct,
            "correct_steps": sum(1 for v in step_details.values() if v["correct"]),
            "total_steps": step_count,
            "step_details": json.dumps(step_details, ensure_ascii=False),
            "prediction": prediction,
            "judge_response": judge_response if judge_response else "",
        })

        judge_completed += 1
        if judge_completed % 20 == 0:
            print(f"  Judge progress: {judge_completed}/{n_eval}", flush=True)

    # ================================================================
    # Phase 4: Compute metrics & save
    # ================================================================
    print(f"\nPhase 4: Computing metrics...")
    n_questions = len(results)
    question_acc = correct_questions / n_questions * 100 if n_questions > 0 else 0
    step_acc = correct_steps / total_steps * 100 if total_steps > 0 else 0

    # Per-task breakdown
    task_stats = defaultdict(lambda: {"q": 0, "cq": 0, "s": 0, "cs": 0})
    for dr in detailed_rows:
        t = dr["task"]
        task_stats[t]["q"] += 1
        task_stats[t]["s"] += dr["total_steps"]
        task_stats[t]["cs"] += dr["correct_steps"]
        if dr["all_correct"]:
            task_stats[t]["cq"] += 1

    # Save
    pred_df = pd.DataFrame(results)
    pred_df.to_excel(OUTPUT_EXCEL, index=False)

    detail_df = pd.DataFrame(detailed_rows)
    detail_df.to_excel(OUTPUT_DETAILS, index=False)

    # Also save a FLAT step-level detail file: one row per step
    flat_rows = []
    for r in results:
        q_id = r["question_id"]
        lang = r["language"]
        task = r["task"]
        prediction = r["prediction"]
        steps = json.loads(r.get("steps_json", "[]"))
        gt_answers = json.loads(r.get("gt_answers", "{}"))
        judge_response = r.get("_judge_response", "")
        # parse judge response for this question
        is_cn = "Chinese" in lang
        step_count = r["step_count"]
        if judge_response:
            pred_answers = parse_judge_response(judge_response, step_count, lang)
        else:
            pred_answers = {}

        for s in steps:
            sn = s["step_num"]
            sk = f"step{sn}"
            gt_step = gt_answers.get(sk, {})
            gt_answer = gt_step.get("answer", "") if isinstance(gt_step, dict) else str(gt_step)
            qa_type = gt_step.get("type", "short_answer") if isinstance(gt_step, dict) else "short_answer"
            pred_answer = pred_answers.get(sk, "")
            step_opts = ""
            for si in steps:
                if si.get('step_num') == sn:
                    step_opts = si.get('qa_options', '')
                    break
            is_correct = check_step_answer(pred_answer, gt_answer, qa_type, step_opts)
            flat_rows.append({
                "question_id": q_id,
                "language": lang,
                "task": task,
                "step_number": sn,
                "total_steps": step_count,
                "qa_question": s["qa_question"],
                "qa_type": qa_type,
                "ground_truth": gt_answer,
                "judge_extracted_answer": pred_answer,
                "is_correct": is_correct,
                "model_raw_prediction": prediction,
                "judge_raw_response": judge_response,
            })

    flat_df = pd.DataFrame(flat_rows)
    flat_output = OUTPUT_DETAILS.replace('_step_details.xlsx', '_step_flat.xlsx')
    flat_df.to_excel(flat_output, index=False)
    print(f"  Flat step details: {flat_output} ({len(flat_rows)} rows)")

    summary_rows = [{
        "Task": "Overall",
        "Questions": n_questions,
        "Correct_Questions": correct_questions,
        "Question_Acc": f"{question_acc:.2f}%",
        "Total_Steps": total_steps,
        "Correct_Steps": correct_steps,
        "Step_Acc": f"{step_acc:.2f}%",
    }]
    for task, stats in sorted(task_stats.items()):
        qa = stats["cq"] / stats["q"] * 100 if stats["q"] > 0 else 0
        sa = stats["cs"] / stats["s"] * 100 if stats["s"] > 0 else 0
        summary_rows.append({
            "Task": task,
            "Questions": stats["q"],
            "Correct_Questions": stats["cq"],
            "Question_Acc": f"{qa:.2f}%",
            "Total_Steps": stats["s"],
            "Correct_Steps": stats["cs"],
            "Step_Acc": f"{sa:.2f}%",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    # Print
    print(f"\n{'='*70}")
    print(f"  Step Reasoning QA — {MODEL_NAME} (GPT-Judge Eval)")
    print(f"{'='*70}")
    print(f"  Question Accuracy: {question_acc:.2f}% ({correct_questions}/{n_questions})")
    print(f"  Step Accuracy:     {step_acc:.2f}% ({correct_steps}/{total_steps})")
    print(f"\n  Per-task breakdown:")
    for task, stats in sorted(task_stats.items()):
        qa = stats["cq"] / stats["q"] * 100 if stats["q"] > 0 else 0
        sa = stats["cs"] / stats["s"] * 100 if stats["s"] > 0 else 0
        print(f"    {task}: Q-Acc={qa:.1f}% ({stats['cq']}/{stats['q']}), "
              f"Step-Acc={sa:.1f}% ({stats['cs']}/{stats['s']})")
    print(f"\n  Output: {OUTPUT_DIR}/")
    print(f"{'='*70}")

    return {"question_acc": question_acc, "step_acc": step_acc}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_evaluation(args.start, args.end)
