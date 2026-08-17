"""
Step-level Reasoning QA Evaluation Utilities (GPT-Judge Version)
=================================================================
Evaluates model responses using GPT judge for answer extraction —
same approach as OCR_Reasoning's OcrR_auxeval.
"""
from ...smp import *
import re
import json
import math
from collections import defaultdict


# ============================================================
# GPT Judge Prompts (few-shot, same style as build_ocrr_gpt4_prompt)
# ============================================================

JUDGE_EXAMPLES_CN = """
示例1:
子问题: 2023年品牌二的收入是多少（单位：十亿元人民币）？
模型回答: 2023年总收入为10.1十亿元人民币，品牌二占比为7%。因此，2023年品牌二的收入 = 10.1 * 7% = 0.707 十亿元人民币。
提取答案: 0.707

示例2:
子问题: 图中用哪种颜色的柱子表示数据的依存度？
模型回答: 根据图例，黑色柱子对应相关数据。经过仔细观察，最终答案是黑色。
提取答案: 黑色

示例3:
子问题: 以下哪组数值与四个年份的数据匹配？
选项: A. [0.4, 0.7, 1.1, 1.2] B. [0.4, 0.7, 1.0, 1.2] C. [0.5, 0.7, 1.1, 1.2] D. [0.4, 0.6, 1.1, 1.2]
模型回答: 2000年对应0.4，2007年对应0.7，2012年对应1.1，2017年对应1.2。与选项A完全匹配。所以正确答案是A。
提取答案: A
"""

JUDGE_EXAMPLES_EN = """
Example 1:
Sub-question: How many days does this campaign last?
Model response: The campaign runs from June 3 to June 22. 22 - 3 = 19, plus 1 since both days are included. So the answer is 20 days.
Extracted answer: 20

Example 2:
Sub-question: What is the next holiday in the poster?
Options: A. Mother's Day B. Christmas C. New Year D. Thanksgiving
Model response: The poster date is October 31st. Mother's Day is in May (past), Christmas is December 25 (upcoming). So the answer is B (Christmas).
Extracted answer: B

Example 3:
Sub-question: How many times does the word 'HOTEL' appear in the poster?
Model response: Looking at the poster text: PALACE HOTEL, HOTEL D'ORANCE, HOTEL KURHAUS, GRAND HOTEL, SAVOY HOTEL, HOTEL RAUCH. That's 6 occurrences.
Extracted answer: 6
"""


def build_judge_prompt_cn(question_text: str, step_questions: list[dict],
                          prediction: str) -> str:
    """Build GPT judge prompt to extract step answers from a verbose prediction (CN)."""
    parts = [
        "请从以下AI助手的回答中，提取每个子问题的最终答案。\n\n",
        "以下是几个提取答案的示例：\n",
        JUDGE_EXAMPLES_CN,
        "\n---\n\n",
        f"原问题: {question_text}\n\n",
        "子问题列表:\n",
    ]
    for s in step_questions:
        sq = s["qa_question"]
        opts = s.get("qa_options", "")
        if opts and opts != "[]":
            try:
                options = json.loads(opts)
                opt_str = " 选项: " + " ".join(
                    f"{o['label']}. {o['text']}" for o in options)
                parts.append(f"  子问题{s['step_num']}: {sq}{opt_str}\n")
            except:
                parts.append(f"  子问题{s['step_num']}: {sq}\n")
        else:
            parts.append(f"  子问题{s['step_num']}: {sq}\n")

    parts.append(f"\nAI助手的回答:\n{prediction}\n\n")
    parts.append(
        "请按以下格式输出每个子问题的提取答案（只输出答案，不要解释）：\n"
        "子问题1: [提取的答案]\n"
        "子问题2: [提取的答案]\n"
        "注意：选择题只输出选项字母（如A），简答题输出最终数值或结果。"
    )
    return "".join(parts)


def build_judge_prompt_en(question_text: str, step_questions: list[dict],
                          prediction: str) -> str:
    """Build GPT judge prompt to extract step answers from a verbose prediction (EN)."""
    parts = [
        "Extract the final answer for each sub-question from the AI assistant's response below.\n\n",
        "Here are some examples of answer extraction:\n\n",
        JUDGE_EXAMPLES_EN,
        "\n---\n\n",
        f"Original question: {question_text}\n\n",
        "Sub-questions:\n",
    ]
    for s in step_questions:
        sq = s["qa_question"]
        opts = s.get("qa_options", "")
        if opts and opts != "[]":
            try:
                options = json.loads(opts)
                opt_str = " Options: " + " ".join(
                    f"{o['label']}. {o['text']}" for o in options)
                parts.append(f"  Sub-question {s['step_num']}: {sq}{opt_str}\n")
            except:
                parts.append(f"  Sub-question {s['step_num']}: {sq}\n")
        else:
            parts.append(f"  Sub-question {s['step_num']}: {sq}\n")

    parts.append(f"\nAI assistant's response:\n{prediction}\n\n")
    parts.append(
        "Output the extracted answer for each sub-question in this format "
        "(output ONLY the answer, no explanation):\n"
        "Sub-question 1: [extracted answer]\n"
        "Sub-question 2: [extracted answer]\n"
        "Note: For multiple-choice, output only the option letter (e.g., A). "
        "For short-answer, output the final value/result."
    )
    return "".join(parts)


def parse_judge_response(judge_response: str, step_count: int,
                         language: str = "English") -> dict:
    """Parse GPT judge's extracted step answers into a dict."""
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
            lines = [l.strip() for l in raw.split('\n') if l.strip()]
            raw = lines[0] if lines else raw
            raw = re.sub(r'\*+', '', raw)
            raw = raw.strip('.,;:;!?*[]()"\'` \t\n\r')
            results[f"step{sn}"] = raw if raw else ""
        else:
            results[f"step{sn}"] = ""

    return results


# ============================================================
# Answer Checking (same as before)
# ============================================================

def check_step_answer(predicted: str, ground_truth: str,
                      qa_type: str = "short_answer",
                      qa_options: str = "") -> bool:
    """Robust answer comparison with format leniency.

    Rules:
    1. Multiple choice: accept letter (A/B/C/D) OR full option content
    2. Numeric: use relaxed tolerance (5%), no strict decimal requirement
    3. Format: strip units, punctuation, whitespace before comparing
    """
    if not predicted:
        return False

    pred = predicted.strip()
    gt = ground_truth.strip()

    pred_clean = pred.strip('.,;:;!?*[]()"\'` \t\n\r')
    gt_clean = gt.strip('.,;:;!?*[]()"\'` \t\n\r')

    # Exact match
    if pred_clean.lower() == gt_clean.lower():
        return True

    # Multiple choice: flexible matching
    if qa_type == "multiple_choice":
        choice_match = re.search(r'\b([A-Da-d])\b', pred_clean)
        if choice_match:
            letter = choice_match.group(1).upper()
            if letter == gt_clean.upper().strip():
                return True
            if qa_options and qa_options != "[]":
                try:
                    options = json.loads(qa_options)
                    for opt in options:
                        olabel = opt.get("label","") if isinstance(opt,dict) else opt.strip()[:1]
                        otext = opt.get("text","") if isinstance(opt,dict) else opt.strip()[2:].strip()
                        if olabel.upper() == letter and otext.lower() == gt_clean.lower():
                            return True
                except: pass
        if qa_options and qa_options != "[]":
            try:
                options = json.loads(qa_options)
                for opt in options:
                    olabel = opt.get("label","") if isinstance(opt,dict) else opt.strip()[:1]
                    otext = opt.get("text","") if isinstance(opt,dict) else opt.strip()[2:].strip()
                    if otext.lower() == pred_clean.lower():
                        if olabel.upper() == gt_clean.upper().strip() or otext.lower() == gt_clean.lower():
                            return True
                    if len(otext) > 10 and len(pred_clean) > 5:
                        pw = set(pred_clean.lower().replace('、',' ').replace(',',' ').split())
                        ow = set(otext.lower().replace('、',' ').replace(',',' ').split())
                        if pw and ow and len(pw & ow)/max(len(pw),len(ow)) > 0.7:
                            if olabel.upper() == gt_clean.upper().strip() or otext.lower() == gt_clean.lower():
                                return True
            except: pass

    # Numeric comparison with relaxed tolerance
    units_cn = ['%','元','十亿元','十亿','万','亿','千','百','克','天','个','份','人','次','种','项','站',
                '人民币','美元','米','厘米','毫米','公里','小时','分钟','秒']
    units_en = ['%','$','¥','dollars','yuan','days','hours','minutes','grams','kg','km','m','cm','mm']
    pred_num = re.sub(r'\s+','',pred_clean)
    gt_num = re.sub(r'\s+','',gt_clean)
    for u in units_cn + units_en + [',']:
        pred_num = pred_num.replace(u,'')
        gt_num = gt_num.replace(u,'')
    try:
        pv = float(pred_num); gv = float(gt_num)
        if math.isclose(pv, gv, rel_tol=0.01, abs_tol=0.01): return True
        if gv != 0 and abs(pv - gv) / abs(gv) < 0.02: return True
    except: pass

    # Unit-stripped string match
    ps, gs = pred_clean, gt_clean
    for u in units_cn + units_en: ps = ps.replace(u,''); gs = gs.replace(u,'')
    if re.sub(r'\s+','',ps).lower() == re.sub(r'\s+','',gs).lower(): return True

    # Normalized string
    if re.sub(r'\s+','',pred_clean).lower() == re.sub(r'\s+','',gt_clean).lower(): return True

    # Substring overlap for long text
    if len(gt_clean) > 20 and len(pred_clean) > 5:
        gw = set(gt_clean.lower().replace('、',' ').replace(',',' ').replace('，',' ').split())
        pw = set(pred_clean.lower().replace('、',' ').replace(',',' ').replace('，',' ').split())
        if gw and pw and len(gw & pw) / len(gw) > 0.6: return True

    return False


# ============================================================
# Main Evaluation (GPT-Judge version)
# ============================================================

def evaluate_step_qa(result_file: str, judge_model=None):
    """Evaluate Step Reasoning QA predictions using GPT judge.

    For each sample, GPT judge extracts all step answers from the model's
    verbose prediction. Each extracted answer is compared against GT.

    Args:
        result_file: path to prediction xlsx (columns: index, question, prediction,
                     language, task, step_count, answer)
        judge_model: a VLMEvalKit judge model instance (from build_judge).
                     If None, uses regex fallback.

    Returns:
        dict with question_acc, step_acc, per_task breakdown
    """
    import openpyxl as _openpyxl

    data = load(result_file)

    # Load step QA data for question text and step info
    step_qa_map = _load_step_qa_data()

    question_results = []
    total_steps = 0
    correct_steps = 0
    correct_questions = 0

    for i in range(len(data)):
        item = data.iloc[i]
        q_id = item['index']
        question_text = item.get('question', '')
        prediction = str(item.get('prediction', ''))
        language = item.get('language', 'English')
        task = item.get('task', '')
        step_count = int(item.get('step_count', 0))

        # Parse GT answers
        gt_json = item.get('answer', '{}')
        try:
            gt_answers = json.loads(gt_json)
        except (json.JSONDecodeError, TypeError):
            gt_answers = {}

        # Get step info for judge prompt
        steps = step_qa_map.get(int(q_id), [])

        # Extract answers: use GPT judge if available, else regex fallback
        if judge_model is not None and steps:
            is_cn = "chinese" in str(language).lower()
            if is_cn:
                judge_prompt = build_judge_prompt_cn(question_text, steps, prediction)
            else:
                judge_prompt = build_judge_prompt_en(question_text, steps, prediction)

            try:
                judge_response = judge_model.generate(judge_prompt)
            except Exception as e:
                judge_response = ""

            pred_answers = parse_judge_response(judge_response, step_count, language)
        else:
            # Fallback: regex parsing (legacy)
            pred_answers = _parse_step_answers_regex(prediction, step_count, language)

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

        question_results.append({
            "index": q_id,
            "task": task,
            "language": language,
            "step_count": step_count,
            "all_correct": all_correct,
            "correct_steps": sum(1 for v in step_details.values() if v["correct"]),
            "total_steps": step_count,
            "step_details": json.dumps(step_details, ensure_ascii=False),
        })

    # Aggregate
    n_questions = len(question_results)
    question_acc = correct_questions / n_questions * 100 if n_questions > 0 else 0
    step_acc = correct_steps / total_steps * 100 if total_steps > 0 else 0

    task_stats = defaultdict(lambda: {"q": 0, "cq": 0, "s": 0, "cs": 0})
    for qr in question_results:
        t = qr["task"]
        task_stats[t]["q"] += 1
        task_stats[t]["s"] += qr["total_steps"]
        task_stats[t]["cs"] += qr["correct_steps"]
        if qr["all_correct"]:
            task_stats[t]["cq"] += 1

    # Build summary
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
    detailed_df = pd.DataFrame(question_results)

    # Save
    detailed_pth = result_file.replace('.xlsx', '_step_details.xlsx')
    summary_pth = result_file.replace('.xlsx', '_step_summary.csv')
    flat_pth = result_file.replace('.xlsx', '_step_flat.xlsx')

    dump(detailed_df, detailed_pth)
    dump(summary_df, summary_pth)

    # Also save flat step-level file
    _save_flat_results(data, question_results, flat_pth)

    print(f"\n{'='*60}")
    print(f"Step Reasoning QA Evaluation (GPT-Judge)")
    print(f"{'='*60}")
    print(f"Question Accuracy: {question_acc:.2f}% ({correct_questions}/{n_questions})")
    print(f"Step Accuracy:     {step_acc:.2f}% ({correct_steps}/{total_steps})")
    for task, stats in sorted(task_stats.items()):
        qa = stats["cq"] / stats["q"] * 100 if stats["q"] > 0 else 0
        sa = stats["cs"] / stats["s"] * 100 if stats["s"] > 0 else 0
        print(f"  {task}: Q-Acc={qa:.1f}%, Step-Acc={sa:.1f}%")
    print(f"{'='*60}")

    return {"question_acc": question_acc, "step_acc": step_acc}


# ============================================================
# Helpers
# ============================================================

_STEP_QA_DATA = None


def _load_step_qa_data():
    """Load step QA pairs from step_qa_pairs.xlsx."""
    global _STEP_QA_DATA
    if _STEP_QA_DATA is not None:
        return _STEP_QA_DATA

    import openpyxl as _xl
    candidates = [
        r'T:\PyCharmProject\OCR-Reasoning\OCR-Reasoning-main\outputs\step_qa\step_qa_pairs.xlsx',
        'outputs/step_qa/step_qa_pairs.xlsx',
    ]
    path = None
    for c in candidates:
        if osp.exists(c):
            path = c
            break
    if path is None:
        _STEP_QA_DATA = {}
        return _STEP_QA_DATA

    wb = _xl.load_workbook(path)
    ws = wb.active
    qa_map = defaultdict(list)
    for row in range(2, ws.max_row + 1):
        qid = int(ws.cell(row, 1).value)
        qa_map[qid].append({
            'step_num': ws.cell(row, 5).value,
            'total_steps': ws.cell(row, 6).value,
            'qa_question': str(ws.cell(row, 9).value),
            'qa_answer': str(ws.cell(row, 10).value),
            'qa_type': str(ws.cell(row, 11).value),
            'qa_options': str(ws.cell(row, 12).value),
        })
    _STEP_QA_DATA = dict(qa_map)
    return _STEP_QA_DATA


def _parse_step_answers_regex(response: str, step_count: int,
                              language: str = "English") -> dict:
    """Legacy regex-based step answer parsing (fallback when no GPT judge)."""
    results = {}
    is_cn = "chinese" in language.lower()
    for sn in range(1, step_count + 1):
        if is_cn:
            patterns = [
                rf'(?:子问题|步骤)\s*{sn}\s*[：:]\s*(.+?)(?=(?:子问题|步骤)\s*{sn+1}\s*[：:]|\Z)',
                rf'(?:子问题|步骤)\s*{sn}\s*[：:]\s*(.+)',
            ]
        else:
            patterns = [
                rf'(?:Sub-question|[Ss]tep)\s*{sn}\s*[：:]\s*(.+?)(?=(?:Sub-question|[Ss]tep)\s*{sn+1}\s*[：:]|\Z)',
                rf'(?:Sub-question|[Ss]tep)\s*{sn}\s*[：:]\s*(.+)',
            ]
        answer = None
        for pat in patterns:
            match = re.search(pat, response, re.DOTALL)
            if match:
                raw = match.group(1).strip()
                raw = raw.split('\n')[0].strip()
                raw = raw.strip('*[]()"\'`')
                if raw:
                    answer = raw
                    break
        results[f"step{sn}"] = answer if answer else ""
    return results


def _save_flat_results(data, question_results, flat_pth):
    """Save a flat step-level detail file (one row per step)."""
    flat_rows = []
    for i in range(len(data)):
        item = data.iloc[i]
        qr = question_results[i]
        q_id = item['index']
        language = item.get('language', 'English')
        task = item.get('task', '')
        prediction = str(item.get('prediction', ''))
        step_count = int(item.get('step_count', 0))
        step_details = json.loads(qr['step_details'])

        steps = _load_step_qa_data().get(int(q_id), [])
        for s in steps:
            sn = s['step_num']
            sk = f"step{sn}"
            sd = step_details.get(sk, {})
            flat_rows.append({
                "question_id": q_id,
                "language": language,
                "task": task,
                "step_number": sn,
                "total_steps": step_count,
                "qa_question": s['qa_question'],
                "qa_type": s['qa_type'],
                "ground_truth": s['qa_answer'],
                "judge_extracted_answer": sd.get('predicted', ''),
                "is_correct": sd.get('correct', False),
                "model_raw_prediction": prediction,
            })

    if flat_rows:
        flat_df = pd.DataFrame(flat_rows)
        dump(flat_df, flat_pth)
