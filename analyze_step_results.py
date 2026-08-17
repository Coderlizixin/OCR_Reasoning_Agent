"""
Deep analysis of Step Reasoning QA results.
Categorizes every error and produces a comprehensive report.
"""
import pandas as pd
import json
import re
import sys
import io
from collections import defaultdict, Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

FLAT_FILE = "outputs/Step_Reasoning_QA/Qwen3.5-9B/Qwen3.5-9B_Step_Reasoning_QA_step_flat.xlsx"
df = pd.read_excel(FLAT_FILE)

# ============================================================
# 1. OVERALL METRICS
# ============================================================
n_total = len(df)
n_correct = df['is_correct'].sum()
n_wrong = n_total - n_correct
step_acc = n_correct / n_total * 100

# Question-level: all steps correct?
q_groups = df.groupby('question_id')
n_questions = len(q_groups)
q_correct = 0
q_partial = 0
q_all_wrong = 0
q_miss_by_one = 0
question_results = []
for qid, group in q_groups:
    total = len(group)
    correct = group['is_correct'].sum()
    all_correct = (correct == total)
    if all_correct:
        q_correct += 1
    elif correct == 0:
        q_all_wrong += 1
    else:
        q_partial += 1
        if correct == total - 1:
            q_miss_by_one += 1
    question_results.append({'qid': qid, 'total': total, 'correct': correct, 'all_correct': all_correct})

q_acc = q_correct / n_questions * 100

print("=" * 70)
print("  STEP REASONING QA — DEEP ANALYSIS")
print("=" * 70)
print(f"\n  Total steps:      {n_total}")
print(f"  Correct steps:    {n_correct} ({step_acc:.1f}%)")
print(f"  Wrong steps:      {n_wrong} ({100-step_acc:.1f}%)")
print(f"\n  Total questions:  {n_questions}")
print(f"  All-correct:      {q_correct} ({q_acc:.1f}%)")
print(f"  Partial-correct:  {q_partial} ({q_partial/n_questions*100:.1f}%)")
print(f"    Miss-by-one:    {q_miss_by_one} ({q_miss_by_one/n_questions*100:.1f}%)")
print(f"  All-wrong:        {q_all_wrong} ({q_all_wrong/n_questions*100:.1f}%)")

# ============================================================
# 2. BY LANGUAGE
# ============================================================
print(f"\n{'─'*50}")
print("  按语言分析")
print(f"{'─'*50}")
for lang in df['language'].unique():
    sub = df[df['language'] == lang]
    acc = sub['is_correct'].mean() * 100
    # Question-level
    qids = sub['question_id'].unique()
    q_all = sum(1 for qid in qids if df[df['question_id']==qid]['is_correct'].all())
    print(f"  {lang}: Step-Acc={acc:.1f}% ({sub['is_correct'].sum()}/{len(sub)}), "
          f"Q-Acc={q_all/len(qids)*100:.1f}% ({q_all}/{len(qids)})")

# ============================================================
# 3. BY QA TYPE
# ============================================================
print(f"\n{'─'*50}")
print("  按QA类型分析")
print(f"{'─'*50}")
for qt in df['qa_type'].unique():
    sub = df[df['qa_type'] == qt]
    acc = sub['is_correct'].mean() * 100
    print(f"  {qt}: {sub['is_correct'].sum()}/{len(sub)} = {acc:.1f}%")

# ============================================================
# 4. BY TASK TYPE
# ============================================================
print(f"\n{'─'*50}")
print("  按任务类型分析")
print(f"{'─'*50}")
for task in sorted(df['task'].unique()):
    sub = df[df['task'] == task]
    acc = sub['is_correct'].mean() * 100
    # Q-level
    qids = sub['question_id'].unique()
    q_all = sum(1 for qid in qids if df[df['question_id']==qid]['is_correct'].all())
    print(f"  {task}: Step-Acc={acc:.1f}% ({sub['is_correct'].sum()}/{len(sub)}), "
          f"Q-Acc={q_all/len(qids)*100:.1f}% ({q_all}/{len(qids)})")

# ============================================================
# 5. BY STEP POSITION
# ============================================================
print(f"\n{'─'*50}")
print("  按步骤位置分析")
print(f"{'─'*50}")
for sn in sorted(df['step_number'].unique()):
    sub = df[df['step_number'] == sn]
    acc = sub['is_correct'].mean() * 100
    print(f"  Step {int(sn)}: {sub['is_correct'].sum()}/{len(sub)} = {acc:.1f}%")

# ============================================================
# 6. ERROR CATEGORIZATION
# ============================================================
print(f"\n{'─'*50}")
print("  错误类型分类（所有 {n_wrong} 个错误步骤）")
print(f"{'─'*50}")

wrong = df[~df['is_correct']].copy()

def classify_error(row):
    gt = str(row['ground_truth']).strip()
    pred = str(row['judge_extracted_answer']).strip()
    qt = row['qa_type']
    raw_pred = str(row.get('model_raw_prediction', '')).strip()

    # 1. Multiple choice: model output value instead of letter
    if qt == 'multiple_choice':
        if pred and not re.match(r'^[A-Da-d]$', pred):
            # Judge extracted specific content, not option letter
            if len(pred) > 2:
                return 'MC_format:模型输出具体内容而非选项字母'
            else:
                return 'MC_other:选项格式问题'
        elif not pred:
            return 'Judge_failed:裁判未能提取答案'
        elif pred.upper() != gt.upper():
            return 'MC_wrong:模型选错了选项'

    # 2. Judge failed to extract
    if not pred or pred.lower() in ('nan', 'none', 'null', ''):
        if raw_pred and len(raw_pred) > 50:
            return 'Judge_failed:裁判未能提取答案'
        else:
            return 'Model_failed:模型未生成有效回答'

    # 3. Model says "can't find" / "not shown"
    if any(kw in pred.lower() for kw in ['无法', '未显示', '未找到', 'not shown', 'not visible', 'cannot', 'not specified', 'not clear', '图片中未']):
        return 'Model_extraction:模型未能从图中提取信息'

    if any(kw in raw_pred.lower()[:500] for kw in ['无法从图中', '图片中未显示', 'not shown in', 'cannot be determined from']):
        return 'Model_extraction:模型未能从图中提取信息'

    # 4. Numeric mismatch (calculation error)
    gt_clean = re.sub(r'[,$¥%元十亿万千兆克天人份次种项站\s]', '', gt)
    pred_clean = re.sub(r'[,$¥%元十亿万千兆克天人份次种项站\s]', '', pred)
    try:
        gtv = float(gt_clean)
        pv = float(pred_clean)
        if abs(gtv - pv) > 0.001:
            return f'Calc_error:数值计算错误(GT={gtv:.2f}, Pred={pv:.2f}, diff={abs(gtv-pv):.2f})'
    except:
        pass

    # 5. String matching: semantically similar but different text
    gt_norm = re.sub(r'\s+', ' ', gt.lower().strip('.,;:!?*[]()"\'` \t\n\r'))
    pred_norm = re.sub(r'\s+', ' ', pred.lower().strip('.,;:!?*[]()"\'` \t\n\r'))

    # Partial overlap check
    gt_words = set(gt_norm.split())
    pred_words = set(pred_norm.split())
    if gt_words and pred_words:
        overlap = len(gt_words & pred_words) / max(len(gt_words), len(pred_words))
        if overlap > 0.5:
            return f'String_partial:部分匹配(overlap={overlap:.0%})'

    # 6. Judge extracted intermediate value instead of final answer
    if gt_clean.replace('.','').replace('-','').isdigit() and pred_clean.replace('.','').replace('-','').isdigit():
        try:
            gtv = float(gt_clean)
            pv = float(pred_clean)
            ratio = pv / gtv if gtv != 0 else 999
            if 0.01 < ratio < 0.99:
                return f'Judge_intermediate:裁判提取了中间值(ratio={ratio:.2f})'
        except:
            pass

    return f'Other:其他原因(GT={gt[:40]}, Pred={pred[:40]})'


# Apply classification
wrong['error_category'] = wrong.apply(classify_error, axis=1)

# Group categories
cat_counts = Counter()
cat_details = defaultdict(list)
for _, row in wrong.iterrows():
    cat = row['error_category'].split(':')[0]
    cat_counts[cat] += 1
    cat_details[cat].append(row)

for cat, count in cat_counts.most_common():
    pct = count / n_wrong * 100
    bar = '█' * int(pct / 2)
    print(f"\n  {cat}: {count} ({pct:.1f}%) {bar}")
    # Show 3 examples
    for row in cat_details[cat][:3]:
        gt = str(row['ground_truth'])[:60]
        pred = str(row['judge_extracted_answer'])[:60]
        print(f"    Q{row['question_id']} S{int(row['step_number'])} [{row['qa_type']}]: "
              f"GT=\"{gt}\" → Judge=\"{pred}\"")

# ============================================================
# 7. IMPROVEMENT POTENTIAL
# ============================================================
print(f"\n{'─'*50}")
print("  改进潜力分析")
print(f"{'─'*50}")

# If we fix MC format matching (accept value when correct)
# For MC questions where model output the right content but not the letter
mc_format_fix = sum(1 for _, row in wrong.iterrows()
                    if row['error_category'].startswith('MC_format'))
# If we use semantic matching for string answers
string_partial_fix = sum(1 for _, row in wrong.iterrows()
                         if row['error_category'].startswith('String_partial'))
# Judge extraction errors
judge_fix = sum(1 for _, row in wrong.iterrows()
                if row['error_category'].startswith('Judge_'))

fixable = mc_format_fix + string_partial_fix + judge_fix
real_errors = n_wrong - fixable
potential_acc = (n_correct + fixable) / n_total * 100

print(f"  MC格式问题（可修复）:    {mc_format_fix}")
print(f"  字符串匹配过严（可修复）: {string_partial_fix}")
print(f"  裁判提取错误（可优化）:   {judge_fix}")
print(f"  模型真实错误（不可修复）: {real_errors}")
print(f"  修复后潜在准确率:         {potential_acc:.1f}%")

# Original OCR-Reasoning accuracy for comparison
print(f"\n  原始OCR-Reasoning准确率:  61.0%")
print(f"  当前Step-Acc:              {step_acc:.1f}%")
print(f"  修复后Step-Acc:            {potential_acc:.1f}%")
