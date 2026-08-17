"""
Build Depended Step QA Pairs (Smart Dependency Detection)
===========================================================
Detects ACTUAL dependencies between steps: step n only includes prior
step results if the step text shows it USES those values as input.

Heuristic:
  - Short numeric answers (<=5 chars like "32", "45"): require the value
    to appear in a computation context (=, +, -, sum) in the current step
  - Long numeric answers (>5 chars like "0.707"): appear in current step
  - Text answers: appear as substring in current step text
"""
import openpyxl
import json
import os
import shutil
import re
from collections import defaultdict
import pandas as pd


INPUT_EXCEL = "outputs/step_qa/step_qa_pairs.xlsx"
OUTPUT_DIR = "outputs/step_qa_depended"
OUTPUT_EXCEL = f"{OUTPUT_DIR}/step_qa_pairs.xlsx"
OUTPUT_TSV = f"{OUTPUT_DIR}/step_reasoning_qa.tsv"
OUTPUT_IMAGES = f"{OUTPUT_DIR}/images"
SOURCE_IMAGES = "outputs/step_qa/images"
FALLBACK_IMAGES = r"C:\Users\19639\LMUData\images\OCR_Reasoning"
ORIG_EXCEL = "outputs/Qwen3.5-9B-OCRAgentV5/T20260730_G533ef742/Qwen3.5-9B-OCRAgentV5_OCR_Reasoning.xlsx"


def is_dependency(prior_answer: str, current_step_text: str,
                  current_qa_question: str) -> bool:
    """Check if current step depends on a prior step's answer.

    Uses the step_text (original reasoning) and qa_question to determine
    whether a prior answer is used as INPUT to the current step's computation,
    not just coincidentally present.
    """
    pa = prior_answer.strip()
    if not pa:
        return False

    combined = current_step_text + " " + current_qa_question
    is_numeric = bool(re.match(r'^-?[\d,.]+$', pa.replace(' ', '')))

    if is_numeric:
        pa_clean = pa.replace(',', '').replace(' ', '')
        pa_num = re.search(r'(\d+\.?\d*)', pa_clean)
        if not pa_num:
            return False
        value = pa_num.group(1)

        # Short numbers (<=5 chars like "32", "45", "20"): require computation context
        if len(value) <= 5:
            # Look for the value after = + - × * / or in sum/difference expressions
            patterns = [
                rf'[=＝+\-×*/]\s*{re.escape(value)}',  # after operator
                rf'{re.escape(value)}\s*[+\-×*/]',      # before operator
                rf'[=＝]\s*.*{re.escape(value)}',        # somewhere after =
                rf'{re.escape(value)}\s*元.*\+\s*',      # in a sum of prices
                rf'\+\s*{re.escape(value)}',             # part of a sum
            ]
            for pat in patterns:
                if re.search(pat, combined):
                    return True
            return False
        else:
            # Long numbers ("0.707", "0.168"): check if they appear
            return value in combined.replace(' ', '')
    else:
        # Text answers: check if a significant portion appears
        # For long text answers (>20 chars), check key parts
        if len(pa) > 30:
            # Extract key tokens (skip common punctuation/stopwords)
            key_parts = [w for w in pa.replace('、', ' ').replace('，', ' ').split()
                        if len(w) > 2]
            matches = sum(1 for kp in key_parts if kp in combined)
            return matches >= min(2, len(key_parts) // 2)
        else:
            return pa.lower() in combined.lower()


def detect_dependencies(steps: list[dict]) -> dict:
    """For each step, determine which prior steps it depends on.

    Returns: dict[step_num] -> list of prior step_nums
    """
    sorted_steps = sorted(steps, key=lambda x: x["step_num"])
    deps = {}

    for i, s in enumerate(sorted_steps):
        sn = s["step_num"]
        deps[sn] = []
        if sn == 1:
            continue

        # Check each prior step
        for ps in sorted_steps[:i]:
            psn = ps["step_num"]
            if is_dependency(ps["qa_answer"], s["step_text"], s["qa_question"]):
                deps[sn].append(psn)

    return deps


def build_depended_qa():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES, exist_ok=True)

    # Load original question info
    wb_orig = openpyxl.load_workbook(ORIG_EXCEL)
    ws_orig = wb_orig.active
    orig_info = {}
    for row in range(2, ws_orig.max_row + 1):
        qid = int(ws_orig.cell(row, 1).value)
        orig_info[qid] = {
            "question": str(ws_orig.cell(row, 2).value),
            "language": str(ws_orig.cell(row, 9).value),
            "task": str(ws_orig.cell(row, 7).value) if ws_orig.cell(row, 7).value else "",
        }

    # Load existing step QA pairs
    wb = openpyxl.load_workbook(INPUT_EXCEL)
    ws = wb.active

    qa_groups = defaultdict(list)
    for row in range(2, ws.max_row + 1):
        qid = int(ws.cell(row, 1).value)
        qa_groups[qid].append({
            "question_id": qid,
            "question": str(ws.cell(row, 2).value),
            "language": str(ws.cell(row, 3).value),
            "full_reasoning": str(ws.cell(row, 4).value),
            "step_num": ws.cell(row, 5).value,
            "total_steps": ws.cell(row, 6).value,
            "step_text": str(ws.cell(row, 7).value),
            "step_types": str(ws.cell(row, 8).value),
            "qa_question": str(ws.cell(row, 9).value),
            "qa_answer": str(ws.cell(row, 10).value),
            "qa_type": str(ws.cell(row, 11).value),
            "qa_options": str(ws.cell(row, 12).value),
            "qa_cot": str(ws.cell(row, 13).value),
            "image_path": str(ws.cell(row, 14).value),
            "generation_method": str(ws.cell(row, 15).value),
        })

    # Build depended QA pairs
    out_wb = openpyxl.Workbook()
    out_ws = out_wb.active
    out_ws.title = "Step QA Pairs (Depended)"
    headers = [
        "question_id", "question", "language", "full_reasoning",
        "step_number", "total_steps", "step_text", "step_types",
        "qa_question", "qa_answer", "qa_type", "qa_options",
        "qa_cot", "image_path", "generation_method", "depends_on"
    ]
    for col, header in enumerate(headers, 1):
        out_ws.cell(1, col, header)

    out_row = 2
    ts_rows = []
    dep_stats = {"total_steps": 0, "steps_with_deps": 0, "total_deps": 0}

    for qid in sorted(qa_groups.keys()):
        steps = sorted(qa_groups[qid], key=lambda x: x["step_num"])
        info = orig_info.get(qid, {})
        lang = info.get("language", "English")
        is_cn = "Chinese" in lang
        total = len(steps)

        # Detect dependencies
        deps = detect_dependencies(steps)

        # Build a lookup: step_num -> qa_answer
        answers_by_step = {s["step_num"]: s["qa_answer"] for s in steps}
        questions_by_step = {s["step_num"]: s["qa_question"] for s in steps}

        for s in steps:
            sn = s["step_num"]
            dep_steps = deps.get(sn, [])
            dep_steps_str = ",".join(str(d) for d in dep_steps) if dep_steps else ""

            if dep_steps:
                dep_stats["steps_with_deps"] += 1
                dep_stats["total_deps"] += len(dep_steps)

            # Keep the original question text — dependencies go into a
            # separate context block, not embedded in the question itself.
            s["qa_question_new"] = s["qa_question"]
            s["depends_on"] = dep_steps_str

            # Copy image
            img_src = os.path.join(SOURCE_IMAGES, f"{qid}.jpg")
            if not os.path.exists(img_src):
                img_src = os.path.join(FALLBACK_IMAGES, f"{qid}.jpg")
            img_dst = os.path.join(OUTPUT_IMAGES, f"{qid}.jpg")
            img_rel = f"images/{qid}.jpg"
            if os.path.exists(img_src) and not os.path.exists(img_dst):
                shutil.copy2(img_src, img_dst)

            # Write to Excel
            row_data = [
                qid, s["question"], s["language"], s["full_reasoning"],
                sn, total, s["step_text"], s["step_types"],
                s["qa_question_new"], s["qa_answer"], s["qa_type"],
                s["qa_options"], s["qa_cot"],
                img_rel, "depended_" + s["generation_method"],
                dep_steps_str
            ]
            for col, value in enumerate(row_data, 1):
                out_ws.cell(out_row, col, value)
            out_row += 1

        # Build TSV row — with dependency context as separate reference block
        updated_steps = sorted(qa_groups[qid], key=lambda x: x["step_num"])

        # Collect steps whose Q&A should appear in the reference block:
        # any step that is a dependency of at least one later step.
        all_depended_steps = set()
        for s in updated_steps:
            for dsn in deps.get(s["step_num"], []):
                all_depended_steps.add(dsn)

        # Build reference block
        ref_parts = []
        if all_depended_steps:
            if is_cn:
                ref_parts.append("以下中间步骤及其答案可供后续步骤参考：")
                for dsn in sorted(all_depended_steps):
                    ref_parts.append(f"步骤{dsn}: {questions_by_step[dsn]}")
                    ref_parts.append(f"答案: {answers_by_step[dsn]}")
                ref_parts.append("")  # blank line separator
            else:
                ref_parts.append("The following intermediate steps and their answers "
                                 "are provided for reference in later steps:")
                for dsn in sorted(all_depended_steps):
                    ref_parts.append(f"Step {dsn}: {questions_by_step[dsn]}")
                    ref_parts.append(f"Answer: {answers_by_step[dsn]}")
                ref_parts.append("")
        ref_block = "\n".join(ref_parts)

        if is_cn:
            parts = ["请根据图片内容，逐步回答以下子问题：\n"]
            for s in updated_steps:
                sn = s["step_num"]
                sq = s["qa_question"]  # original question, unchanged
                st = s["qa_type"]
                opts = s["qa_options"]
                if st == "multiple_choice" and opts and opts != "[]":
                    try:
                        options = json.loads(opts)
                        opt_text = "\n".join(f"{o['label']}. {o['text']}" for o in options)
                        parts.append(f"子问题{sn}: {sq}\n选项:\n{opt_text}")
                    except:
                        parts.append(f"子问题{sn}: {sq}")
                else:
                    parts.append(f"子问题{sn}: {sq}")
            parts.append(
                "\n按顺序输出每个子问题的答案。答案格式："
                "\n子问题1: [你的答案]\n子问题2: [你的答案]\n...\n"
                "注意：选择题只输出选项字母（A/B/C/D），简答题直接输出数值或结果。"
            )
        else:
            parts = ["Please answer the following sub-questions step by step based on the image:\n"]
            for s in updated_steps:
                sn = s["step_num"]
                sq = s["qa_question"]  # original question, unchanged
                st = s["qa_type"]
                opts = s["qa_options"]
                if st == "multiple_choice" and opts and opts != "[]":
                    try:
                        options = json.loads(opts)
                        opt_text = "\n".join(f"{o['label']}. {o['text']}" for o in options)
                        parts.append(f"Sub-question {sn}: {sq}\nOptions:\n{opt_text}")
                    except:
                        parts.append(f"Sub-question {sn}: {sq}")
                else:
                    parts.append(f"Sub-question {sn}: {sq}")
            parts.append(
                "\nOutput your answer for each sub-question in order. Answer format:"
                "\nSub-question 1: [your answer]\nSub-question 2: [your answer]\n...\n"
                "Note: For multiple-choice, output only the option letter (A/B/C/D). "
                "For short-answer, output the value or result directly."
            )

        # Assemble: reference block (if any) + step questions
        question_text = "\n".join(parts)
        if ref_block:
            question_text = ref_block + "\n" + question_text

        full_prompt = question_text
        gt_answers = {}
        for s in updated_steps:
            gt_answers[f"step{s['step_num']}"] = {
                "answer": s["qa_answer"], "type": s["qa_type"],
            }

        img_path = os.path.abspath(os.path.join(OUTPUT_IMAGES, f"{qid}.jpg"))
        if not os.path.exists(img_path):
            img_path = os.path.abspath(os.path.join(FALLBACK_IMAGES, f"{qid}.jpg"))

        ts_rows.append({
            "index": qid, "question": full_prompt,
            "answer": json.dumps(gt_answers, ensure_ascii=False),
            "image": img_path, "language": lang,
            "task": info.get("task", ""), "step_count": total,
            "orig_question": info.get("question", ""),
        })

    out_wb.save(OUTPUT_EXCEL)
    df_tsv = pd.DataFrame(ts_rows)
    df_tsv.to_csv(OUTPUT_TSV, sep="\t", index=False)

    # Summary
    dep_stats["total_steps"] = out_row - 2
    print(f"Depended Step QA built!")
    print(f"  Total steps: {dep_stats['total_steps']}")
    print(f"  Steps with dependencies: {dep_stats['steps_with_deps']} "
          f"({dep_stats['steps_with_deps']/dep_stats['total_steps']*100:.1f}%)")
    print(f"  Total dependency edges: {dep_stats['total_deps']}")
    print(f"  Output: {OUTPUT_DIR}/")
    print()

    # Examples
    for qid in [13, 60]:
        steps = sorted(qa_groups[qid], key=lambda x: x["step_num"])
        deps = detect_dependencies(steps)
        print(f"=== Q{qid} dependency graph ===")
        for s in steps:
            sn = s["step_num"]
            dep = deps.get(sn, [])
            a = s["qa_answer"]
            dep_str = f" ← depends on steps {dep}" if dep else " (no deps)"
            print(f"  Step{sn} [{a[:40]}]{dep_str}")
        print()


if __name__ == "__main__":
    build_depended_qa()
