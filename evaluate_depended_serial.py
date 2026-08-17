"""
Depended Step QA — Serial Evaluation
======================================
For each question, calls the model SERIALLY step by step.
Step n receives the model's ACTUAL answers to steps 1..n-1 as context
(not GT answers), simulating a real chain-of-thought dependency.

Output format matches VLMEvalKit prediction xlsx, so it can be directly
evaluated via Depended_Step_Reasoning_QA.evaluate().
"""
import openpyxl
import json
import os
import sys
import re
import time
import base64
import math
from collections import defaultdict
from typing import Optional
import pandas as pd


# ============================================================
# CONFIG — same as Qwen3.5-9B-API
# ============================================================
MODEL_NAME = "Qwen3.5-4B"
API_BASE = "http://10.3.112.34:23200/v1/chat/completions"
API_KEY = "EMPTY"
MAX_TOKENS = 6144  # per-step call
TEMPERATURE = 0.6
IMG_DETAIL = "high"
RETRY = 3

# GPT Judge
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

JUDGE_MODEL = "qwen-plus"
JUDGE_API_KEY = os.environ.get("OPENAI_API_KEY", "")
JUDGE_API_BASE = os.environ.get("OPENAI_API_BASE",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

# I/O
DEPENDED_TSV = "outputs/step_qa_depended/step_reasoning_qa.tsv"  # 40 questions
DEPENDED_EXCEL = "outputs/step_qa_depended/step_qa_pairs.xlsx"  # step details
IMAGE_DIR = "outputs/step_qa/images"
FALLBACK_IMAGES = r"C:/Users/19639/LMUData/images/OCR_Reasoning"
OUTPUT_DIR = f"outputs/Depended_Step_Reasoning_QA_Think_4B/{MODEL_NAME}"
OUTPUT_PRED = f"{OUTPUT_DIR}/{MODEL_NAME}_Depended_Step_Reasoning_QA_Think.xlsx"

# ============================================================
# API CALLER
# ============================================================
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_api(messages: list[dict], max_tokens: int = MAX_TOKENS,
             model: str = None, api_base: str = None, api_key: str = None) -> Optional[str]:
    import urllib.request
    model_name = model or MODEL_NAME
    endpoint = api_base or API_BASE
    auth_key = api_key or API_KEY
    payload = json.dumps({
        "model": model_name, "messages": messages,
        "temperature": TEMPERATURE, "max_tokens": max_tokens,
        "top_k": 20,
    }, ensure_ascii=False).encode('utf-8')
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {auth_key}"}
    for attempt in range(RETRY):
        try:
            req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode('utf-8'))["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == RETRY - 1:
                return f"[API_ERROR: {e}]"
            time.sleep((attempt + 1) * 3)

# ============================================================
# GPT JUDGE EXTRACTION (extract answer from thinking-mode response)
# ============================================================
def judge_extract(model_response: str, language: str, question: str) -> str:
    """Use GPT judge to extract the final answer from a verbose thinking response.

    The judge reads the model's thinking output and extracts just the answer.
    """
    is_cn = "chinese" in language.lower()
    if is_cn:
        judge_prompt = (
            "从以下AI助手的回答中提取子问题的最终答案。只输出答案值本身，不要解释。\n\n"
            f"子问题: {question}\n\n"
            f"AI助手的回答:\n{model_response}\n\n"
            "提取的答案:"
        )
    else:
        judge_prompt = (
            "Extract the final answer to the sub-question from the AI assistant's response below. "
            "Output ONLY the answer value, no explanation.\n\n"
            f"Sub-question: {question}\n\n"
            f"AI assistant's response:\n{model_response}\n\n"
            "Extracted answer:"
        )

    judge_msgs = [{"role": "user", "content": judge_prompt}]
    judge_resp = call_api(judge_msgs, max_tokens=256,
                          model=JUDGE_MODEL, api_base=JUDGE_API_BASE, api_key=JUDGE_API_KEY)
    if judge_resp:
        return judge_resp.strip().strip('.,;:;!?*[]()"\'` \t\n\r')
    return ""


# ============================================================
# STEP ANSWER PARSER (fallback regex extraction)
# ============================================================
def extract_step_answer(response: str, language: str) -> str:
    """Extract the final answer from <a>answer</a> tags in model response."""
    if not response:
        return ""

    # Priority 1: <a>...</a> tags (anywhere in response, including inside thinking)
    match = re.search(r'<a>\s*(.+?)\s*</a>', response, re.IGNORECASE | re.DOTALL)
    if match:
        raw = match.group(1).strip()
        raw = raw.split('\n')[0].strip()  # first line only
        raw = re.sub(r'\*+', '', raw)
        raw = raw.strip('.,;:;!?*[]()"\'` \t\n\r')
        if raw and len(raw) < 200:
            return raw

    # Priority 2: "答案:" / "Answer:" patterns
    for pat in [r'答案[：:]\s*(.+?)(?:\n|$)', r'Answer[：:]\s*(.+?)(?:\n|$)']:
        match = re.search(pat, response, re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            raw = re.sub(r'\*+', '', raw)
            raw = raw.strip('.,;:;!?*[]()"\'` \t\n\r')
            if raw and len(raw) < 200:
                return raw

    # Priority 3: last short meaningful line
    lines = [l.strip() for l in response.split('\n') if l.strip()
             and len(l.strip()) < 100 and not l.strip().startswith('```')]
    for l in reversed(lines):
        l = re.sub(r'\*+', '', l)
        l = l.strip('.,;:;!?*[]()"\'` \t\n\r')
        if l:
            return l
    return ""

# ============================================================
# PROMPT BUILDER — single step, with prior model answers
# ============================================================
def get_option_text(letter: str, options_str: str) -> str:
    """Get full option text for a letter."""
    if not options_str or not letter:
        return letter
    try:
        for opt in json.loads(options_str):
            if isinstance(opt, str) and opt.strip().startswith(letter.strip().upper() + "."):
                return opt.strip()
            if isinstance(opt, dict) and opt.get("label", "").upper() == letter.strip().upper():
                return f"{letter.strip().upper()}. {opt.get('text', '')}"
    except:
        pass
    return letter

def build_step_prompt(step: dict, prior_answers: list[dict],
                      image_path: str, language: str,
                      original_question: str = "", original_format: str = "") -> list[dict]:
    """Build messages for a single step — OCR-Reasoning style with <a> tags."""
    is_cn = "chinese" in language.lower()
    b64 = encode_image(image_path)
    data_url = f"data:image/jpeg;base64,{b64}"

    sq = step["qa_question"]
    st = step.get("qa_type", "short_answer")
    opts = step.get("qa_options", "[]")

    if is_cn:
        text = f"通过一步一步的推理解决以下问题。最后用 <a>答案</a> 标签包围最终答案。\n问题: <q>{sq}</q>"
        if st == "multiple_choice" and opts and opts != "[]":
            try:
                options = json.loads(opts)
                opt_lines = [o if isinstance(o, str) else f"{o['label']}. {o['text']}" for o in options]
                text += f"\n选项:\n" + "\n".join(opt_lines)
                text += f"\n注意：只需输出选项字母，如 <a>A</a>。"
            except:
                pass
    else:
        text = f"Solve the following question through step-by-step reasoning. Enclose your final answer within <a> tags.\nQuestion: <q>{sq}</q>"
        if st == "multiple_choice" and opts and opts != "[]":
            try:
                options = json.loads(opts)
                opt_lines = [o if isinstance(o, str) else f"{o['label']}. {o['text']}" for o in options]
                text += f"\nOptions:\n" + "\n".join(opt_lines)
                text += f"\nNote: output only the option letter, e.g., <a>A</a>."
            except:
                pass

    return [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url, "detail": IMG_DETAIL}},
            {"type": "text", "text": text},
        ]}
    ]

# ============================================================
# SERIAL INFERENCE
# ============================================================
def run_serial_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load depended QA data (step-level, with depends_on column)
    wb = openpyxl.load_workbook(DEPENDED_EXCEL)
    ws = wb.active
    qa_groups = defaultdict(list)
    for row in range(2, ws.max_row + 1):
        qid = int(ws.cell(row, 1).value)
        qa_groups[qid].append({
            "question_id": qid,
            "question": str(ws.cell(row, 2).value),
            "language": str(ws.cell(row, 3).value),
            "step_num": ws.cell(row, 5).value,
            "total_steps": ws.cell(row, 6).value,
            "qa_question": str(ws.cell(row, 9).value),
            "qa_answer": str(ws.cell(row, 10).value),
            "qa_type": str(ws.cell(row, 11).value),
            "qa_options": str(ws.cell(row, 12).value),
            "depends_on": str(ws.cell(row, 16).value) if ws.cell(row, 16).value else "",
        })

    # Load original info
    wb_orig = openpyxl.load_workbook(
        "outputs/Qwen3.5-9B-OCRAgentV5/T20260730_G533ef742/Qwen3.5-9B-OCRAgentV5_OCR_Reasoning.xlsx"
    )
    ws_orig = wb_orig.active
    orig_info = {}
    for row in range(2, ws_orig.max_row + 1):
        qid = int(ws_orig.cell(row, 1).value)
        orig_info[qid] = {
            "question": str(ws_orig.cell(row, 2).value),
            "language": str(ws_orig.cell(row, 9).value),
            "format": str(ws_orig.cell(row, 8).value) if ws_orig.cell(row, 8).value else "",
            "task": str(ws_orig.cell(row, 7).value) if ws_orig.cell(row, 7).value else "",
        }

    n_questions = len(qa_groups)
    # Filter to only the first 40 questions (those in the TSV)
    tsv_df = pd.read_csv(DEPENDED_TSV, sep="\t")
    tsv_qids = set(int(x) for x in tsv_df["index"])
    qa_groups = {qid: steps for qid, steps in qa_groups.items() if qid in tsv_qids}

    print(f"After TSV filter: {len(qa_groups)} questions, "
          f"{sum(len(v) for v in qa_groups.values())} steps")
    print(f"Mode: SERIAL (model's own answers used as step dependencies)\n")

    results = []

    for qi, qid in enumerate(sorted(qa_groups.keys())):
        steps = sorted(qa_groups[qid], key=lambda x: x["step_num"])
        info = orig_info.get(qid, {})
        lang = info.get("language", "English")
        total = len(steps)

        # Image
        img_path = os.path.join(IMAGE_DIR, f"{qid}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(FALLBACK_IMAGES, f"{qid}.jpg")

        if not os.path.exists(img_path):
            print(f"  Q{qid}: image not found, skipping")
            continue

        # Serial inference: step by step
        prior_answers = []  # list of {step_num, question, answer}
        step_responses = []  # per-step raw model responses

        for s in steps:
            sn = s["step_num"]
            msgs = build_step_prompt(s, prior_answers, img_path, lang,
                                     info.get("question", ""),
                                     info.get("format", ""))
            response = call_api(msgs)

            # Use GPT judge to extract answer from thinking-mode response
            model_answer = judge_extract(response, lang, s["qa_question"]) if response else ""

            step_responses.append({
                "step_num": sn,
                "raw_response": response if response else "",
                "extracted_answer": model_answer,
                "prompt_context_count": len(prior_answers),
            })

            # Add to prior answers for next step — use model's actual output
            prior_answers.append({
                "step_num": sn,
                "question": s["qa_question"],
                "answer": model_answer,
            })

            if qi < 3:  # verbose for first 3 questions
                print(f"  Q{qid} Step{sn}/{total}: answer=\"{model_answer[:60]}\"")

        # Build GT answers and prediction text (full concatenation for judge)
        gt_answers = {}
        for s in steps:
            gt_answers[f"step{s['step_num']}"] = {
                "answer": s["qa_answer"], "type": s["qa_type"],
            }

        prediction_parts = []
        for sr in step_responses:
            prediction_parts.append(
                f"[Step {sr['step_num']}]\n"
                f"Model raw response:\n{sr['raw_response']}\n"
                f"Extracted answer: {sr['extracted_answer']}"
            )
        prediction_text = "\n\n".join(prediction_parts)

        # Build question text (for evaluation — the step questions combined)
        is_cn = "Chinese" in str(lang)
        if is_cn:
            q_lines = ["通过一步一步的推理回答以下子问题：\n"]
            for s in steps:
                q_lines.append(f"子问题{s['step_num']}: {s['qa_question']}")
            question_text = "\n".join(q_lines)
        else:
            q_lines = ["Answer the following sub-questions through step-by-step reasoning:\n"]
            for s in steps:
                q_lines.append(f"Sub-question {s['step_num']}: {s['qa_question']}")
            question_text = "\n".join(q_lines)

        results.append({
            "index": qid,
            "question": question_text,
            "answer": json.dumps(gt_answers, ensure_ascii=False),
            "prediction": prediction_text,
            "language": lang,
            "task": info.get("task", ""),
            "step_count": total,
            "_step_responses": json.dumps(step_responses, ensure_ascii=False),
        })

        if (qi + 1) % 10 == 0:
            print(f"  Progress: {qi + 1}/{n_questions} questions done")

    # Save predictions in VLMEvalKit-compatible format
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_PRED, index=False)
    print(f"\nPredictions saved: {OUTPUT_PRED}")
    print(f"  Questions: {len(df)}, Steps: {df['step_count'].sum()}")

    # ============================================================
    # Evaluate using GPT judge (same as Step_Reasoning_QA.evaluate)
    # ============================================================
    print(f"\nRunning GPT-judge evaluation ({JUDGE_MODEL})...")

    # Inline evaluation functions (avoid VLMEvalKit dependency)
    from evaluate_step_reasoning_qa import (
        build_judge_prompt_cn, build_judge_prompt_en,
        parse_judge_response, check_step_answer,
    )

    # Load step QA data for judge prompts
    step_qa_map = {}
    wb2 = openpyxl.load_workbook(DEPENDED_EXCEL)
    ws2 = wb2.active
    for row in range(2, ws2.max_row + 1):
        qid = int(ws2.cell(row, 1).value)
        if qid not in step_qa_map:
            step_qa_map[qid] = []
        step_qa_map[qid].append({
            'step_num': ws2.cell(row, 5).value,
            'qa_question': str(ws2.cell(row, 9).value),
            'qa_answer': str(ws2.cell(row, 10).value),
            'qa_type': str(ws2.cell(row, 11).value),
            'qa_options': str(ws2.cell(row, 12).value),
        })

    # Load original OCR_Reasoning question answers for Q-Acc
    orig_wb = openpyxl.load_workbook(
        "outputs/Qwen3.5-9B-OCRAgentV5/T20260730_G533ef742/Qwen3.5-9B-OCRAgentV5_OCR_Reasoning.xlsx"
    )
    orig_ws = orig_wb.active
    orig_answers = {}
    for row in range(2, orig_ws.max_row + 1):
        qid = int(orig_ws.cell(row, 1).value)
        orig_answers[qid] = {
            "answer": str(orig_ws.cell(row, 5).value),  # final answer
            "answer_type": str(orig_ws.cell(row, 6).value) if orig_ws.cell(row, 6).value else "str",
        }

    data = pd.read_excel(OUTPUT_PRED)

    question_results = []
    total_steps = correct_steps = 0
    correct_questions_last = 0   # Q-Acc: last step correct
    total_dep_steps = correct_dep_steps = 0  # Depended Acc
    correct_dr_steps = 0  # D-R Acc

    for i in range(len(data)):
        item = data.iloc[i]
        q_id = item['index']
        question_text = item.get('question', '')
        prediction = str(item.get('prediction', ''))
        language = item.get('language', 'English')
        task = item.get('task', '')
        step_count = int(item.get('step_count', 0))

        gt_json = item.get('answer', '{}')
        try:
            gt_answers = json.loads(gt_json)
        except:
            gt_answers = {}

        steps_info = step_qa_map.get(int(q_id), [])

        is_cn = "chinese" in str(language).lower()
        if is_cn:
            judge_prompt = build_judge_prompt_cn(question_text, steps_info, prediction)
        else:
            judge_prompt = build_judge_prompt_en(question_text, steps_info, prediction)

        judge_msgs = [{"role": "user", "content": judge_prompt}]
        judge_response = call_api(judge_msgs, max_tokens=1024,
                                  model=JUDGE_MODEL,
                                  api_base=JUDGE_API_BASE,
                                  api_key=JUDGE_API_KEY)

        if judge_response:
            pred_answers = parse_judge_response(judge_response, step_count, language)
        else:
            pred_answers = {}

        all_correct = True
        step_details = {}
        for sn in range(1, step_count + 1):
            sk = f"step{sn}"
            gt_step = gt_answers.get(sk, {})
            gt_a = gt_step.get("answer", "") if isinstance(gt_step, dict) else str(gt_step)
            qa_t = gt_step.get("type", "short_answer") if isinstance(gt_step, dict) else "short_answer"
            pred_a = pred_answers.get(sk, "")
            # Get step options for robust MC matching
            step_opts = ""
            for si in steps_info:
                if si['step_num'] == sn:
                    step_opts = si.get('qa_options', '')
                    break
            is_c = check_step_answer(pred_a, gt_a, qa_t, step_opts)
            total_steps += 1
            if is_c:
                correct_steps += 1
            else:
                all_correct = False

            # Track dependent steps for R-Q Acc
            dep_on = ""
            for si in steps_info:
                if si['step_num'] == sn:
                    dep_on = si.get('depends_on', '')
                    break
            is_dep_step = bool(dep_on and dep_on.strip())
            if is_dep_step:
                total_dep_steps += 1
                if is_c:
                    correct_dep_steps += 1
                # D-R Acc: this step AND all its prerequisite steps correct
                prereq_nums = [int(x.strip()) for x in dep_on.split(',') if x.strip()]
                all_prereq_ok = True
                for pn in prereq_nums:
                    pk = f"step{pn}"
                    if pk in step_details and not step_details[pk]["correct"]:
                        all_prereq_ok = False
                        break
                if is_c and all_prereq_ok:
                    correct_dr_steps += 1

            step_details[sk] = {"predicted": pred_a, "ground_truth": gt_a,
                                "correct": is_c, "type": qa_t}

        # Q-Acc: last step answer matches the original question answer
        last_correct = False
        if step_count > 0:
            last_sk = f"step{step_count}"
            last_pred = pred_answers.get(last_sk, "")
            orig_ans = orig_answers.get(int(q_id), {})
            orig_answer = orig_ans.get("answer", "")
            orig_type = orig_ans.get("answer_type", "str")
            # Get last step's MC options for robust matching
            last_opts = ""
            for si in steps_info:
                if si['step_num'] == step_count:
                    last_opts = si.get('qa_options', '')
                    break
            last_correct = check_step_answer(last_pred, orig_answer, orig_type, last_opts)
            if last_correct:
                correct_questions_last += 1

        question_results.append({
            "index": q_id, "task": task, "language": language,
            "step_count": step_count,
            "R_Q_correct": all_correct,      # all steps correct
            "Q_correct": last_correct,        # last step matches original answer
            "correct_steps": sum(1 for v in step_details.values() if v["correct"]),
            "total_steps": step_count,
            "step_details": json.dumps(step_details, ensure_ascii=False),
        })

    n_q = len(results)
    dep_acc = correct_dep_steps / total_dep_steps * 100 if total_dep_steps > 0 else 0
    dr_acc = correct_dr_steps / total_dep_steps * 100 if total_dep_steps > 0 else 0
    q_acc = correct_questions_last / n_q * 100 if n_q > 0 else 0
    s_acc = correct_steps / total_steps * 100 if total_steps > 0 else 0

    # Save
    detailed_pth = OUTPUT_PRED.replace('.xlsx', '_step_details.xlsx')
    summary_pth = OUTPUT_PRED.replace('.xlsx', '_step_summary.csv')
    flat_pth = OUTPUT_PRED.replace('.xlsx', '_step_flat.xlsx')

    detail_df = pd.DataFrame(question_results)
    detail_df.to_excel(detailed_pth, index=False)

    summary_rows = [{"Task": "Overall", "Questions": n_q,
                     "Q_Acc": f"{q_acc:.2f}%",
                     "Q_Correct": correct_questions_last,
                     "Depended_Acc": f"{dep_acc:.2f}%",
                     "D_R_Acc": f"{dr_acc:.2f}%",
                     "Dep_Steps": total_dep_steps,
                     "Dep_Correct": correct_dep_steps,
                     "DR_Correct": correct_dr_steps,
                     "Step_Acc": f"{s_acc:.2f}%",
                     "Total_Steps": total_steps,
                     "Correct_Steps": correct_steps}]
    pd.DataFrame(summary_rows).to_csv(summary_pth, index=False)

    # Flat: one row per step, with per-step raw model response
    flat_rows = []
    for i in range(len(data)):
        item = data.iloc[i]
        qr = question_results[i]
        q_id = item['index']
        lang = item.get('language', 'English')
        task = item.get('task', '')
        step_count = int(item.get('step_count', 0))
        step_details = json.loads(qr['step_details'])
        # Get per-step raw responses
        step_responses_json = item.get('_step_responses', '[]')
        try:
            step_responses = json.loads(str(step_responses_json))
        except:
            step_responses = []
        step_response_map = {}
        for sr in step_responses:
            if isinstance(sr, dict):
                step_response_map[sr.get('step_num', 0)] = sr.get('raw_response', '')

        for sn in range(1, step_count + 1):
            sk = f"step{sn}"
            sd = step_details.get(sk, {})
            # Get step question
            steps_info = step_qa_map.get(int(q_id), [])
            sq_text = ""
            for si in steps_info:
                if si['step_num'] == sn:
                    sq_text = si['qa_question']
                    break
            is_last = (sn == step_count)
            flat_rows.append({
                "question_id": q_id, "language": lang, "task": task,
                "step_number": sn, "total_steps": step_count,
                "is_last_step": is_last,
                "qa_question": sq_text,
                "qa_type": sd.get('type', ''),
                "ground_truth": sd.get('ground_truth', ''),
                "judge_extracted_answer": sd.get('predicted', ''),
                "is_correct": sd.get('correct', False),
                "R_Q_correct": qr.get('R_Q_correct', False),
                "Q_correct": qr.get('Q_correct', False),
                "model_raw_prediction": step_response_map.get(sn, ''),
            })
    pd.DataFrame(flat_rows).to_excel(flat_pth, index=False)

    print(f"\n{'='*60}")
    print(f"  Depended Step QA — Serial Mode ({MODEL_NAME})")
    print(f"{'='*60}")
    print(f"  Q-Acc (last step correct):        {q_acc:.2f}% ({correct_questions_last}/{n_q})")
    print(f"  Depended Acc (dependent steps correct): {dep_acc:.2f}% ({correct_dep_steps}/{total_dep_steps})")
    print(f"  D-R Acc (dep. step + all prereqs correct): {dr_acc:.2f}% ({correct_dr_steps}/{total_dep_steps})")
    print(f"  Step Accuracy:                     {s_acc:.2f}% ({correct_steps}/{total_steps})")
    print(f"\n  Output: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_serial_evaluation()
