#!/usr/bin/env python3
"""Comprehensive error analysis for OCRAgentV5 (no decomposition) experiment results."""
import sys
import io
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import json
import re
from collections import Counter, defaultdict

# ── Load data ──────────────────────────────────────────────────────────────
XLSX = r"T:\PyCharmProject\OCR-Reasoning\OCR-Reasoning-main\outputs\Qwen3.5-9B-OCRAgentV5\T20260730_G533ef742\Qwen3.5-9B-OCRAgentV5_OCR_Reasoning_qwen-plus.xlsx"
df = pd.read_excel(XLSX)
print(f"Loaded {len(df)} samples")
print(f"Columns: {list(df.columns)}")
print()

# ── Parse the 'log' column for traces ──────────────────────────────────────
def parse_log(log_str):
    """Parse log column to extract trace info."""
    if pd.isna(log_str):
        return {}
    s = str(log_str)
    info = {}
    # Check status
    if 'planning_parse_failed' in s:
        info['status'] = 'planning_parse_failed'
    elif 'planning_api_failed' in s:
        info['status'] = 'planning_api_failed'
    elif 'execution_error' in s:
        info['status'] = 'execution_error'
    elif 'reasoning_failed' in s or 'reasoning_error' in s:
        info['status'] = 'reasoning_failed'
    elif 'planning_decompose_failed' in s:
        info['status'] = 'planning_decompose_failed'
    elif 'Succeed' in s or 'success' in s.lower():
        info['status'] = 'success'
    else:
        info['status'] = 'unknown'

    # Count attempts
    attempts_match = re.findall(r'attempt[s]?\s*[=:]\s*(\d+)', s, re.IGNORECASE)
    if attempts_match:
        info['attempts'] = int(attempts_match[-1])

    # Planning parse attempts
    ppf_count = s.count('planning_parse_failed')
    info['ppf_count'] = ppf_count

    # Check if trace exists (JSON trace)
    trace_match = re.search(r'"trace"\s*:\s*(\{.*?\})\s*[,}]', s)
    if not trace_match:
        trace_match = re.search(r'"overall_status"\s*:\s*"([^"]+)"', s)
        if trace_match:
            info['trace_status'] = trace_match.group(1)

    return info

# ── Analyze prediction column ──────────────────────────────────────────────
def analyze_prediction(pred):
    """Analyze model prediction for error patterns."""
    if pd.isna(pred):
        return {'has_answer_tag': False, 'pred_len': 0, 'error_type': 'empty_prediction'}

    s = str(pred)
    info = {'pred_len': len(s)}

    # Check for <a> tag
    a_match = re.search(r'<a>(.*?)</a>', s, re.DOTALL)
    info['has_answer_tag'] = bool(a_match)
    if a_match:
        info['extracted_answer'] = a_match.group(1).strip()
    else:
        info['extracted_answer'] = None

    # Check for JSON output
    json_match = re.search(r'\{[^}]*"reasoning"[^}]*"answer"[^}]*\}', s, re.DOTALL)
    info['has_json'] = bool(json_match)

    # Check for truncation (no </a> but has content near end)
    if not info['has_answer_tag'] and len(s) > 1000:
        last_200 = s[-200:]
        if any(kw in last_200.lower() for kw in ['wait', 'however', 'but', 'let me', 'actually', 'alternatively']):
            info['likely_truncated'] = True
        else:
            info['likely_truncated'] = 'maybe'
    else:
        info['likely_truncated'] = False

    # Self-doubt patterns
    doubt_keywords = ['wait', 'however', 'but actually', 'let me reconsider', 'on second thought',
                      'alternatively', 'hmm', 'actually', 'no,', 'i mean', 'rather than']
    doubt_count = sum(s.lower().count(kw) for kw in doubt_keywords)
    info['doubt_count'] = doubt_count

    # Repetitive patterns
    if info['has_answer_tag']:
        # Check for repetitive structure tags
        rep_patterns = ['**', '##', '---', '```']
        rep_count = sum(s.count(p) for p in rep_patterns)
        info['formatting_density'] = rep_count / max(len(s), 1) * 1000

    # Extract answer from prediction (for comparison with res)
    # Try multiple patterns
    answer_patterns = [
        r'<a>\s*(.*?)\s*</a>',
        r'"answer"\s*:\s*"(.*?)"',
        r'"answer"\s*:\s*"([^"]*)"',
    ]
    for pat in answer_patterns:
        m = re.search(pat, s, re.DOTALL | re.IGNORECASE)
        if m:
            info['pred_answer'] = m.group(1).strip()
            break

    if 'pred_answer' not in info:
        info['pred_answer'] = None

    return info

# ── Compare prediction answer vs res (judge-extracted answer) ──────────────
def compare_answers(pred_answer, res):
    """Compare model answer with judge-extracted answer."""
    if pred_answer is None or pd.isna(res):
        return 'no_comparison_possible'
    pred_str = str(pred_answer).strip().lower()
    res_str = str(res).strip().lower()
    if pred_str == res_str:
        return 'exact_match'
    # Check if one contains the other
    if pred_str and res_str:
        if pred_str in res_str or res_str in pred_str:
            return 'partial_match'
    return 'mismatch'

# ── Categorize error types ─────────────────────────────────────────────────
def categorize_error(row, pred_info, log_info):
    """Categorize the error for a failed sample."""
    gt_answer = str(row.get('answer', '')) if not pd.isna(row.get('answer')) else ''
    res_val = str(row.get('res', '')) if not pd.isna(row.get('res')) else ''
    pred = str(row.get('prediction', '')) if not pd.isna(row.get('prediction')) else ''
    reason_score = row.get('reason_score', 0) if not pd.isna(row.get('reason_score', 0)) else 0

    errors = []

    # 1. Pipeline failure errors
    status = log_info.get('status', 'unknown')
    if status == 'planning_parse_failed':
        errors.append('pipeline:planning_parse_failed')
    elif status == 'planning_api_failed':
        errors.append('pipeline:planning_api_failed')
    elif status == 'execution_error':
        errors.append('pipeline:execution_error')
    elif status == 'reasoning_failed':
        errors.append('pipeline:reasoning_failed')

    # 2. Output truncation
    if pred_info.get('likely_truncated'):
        errors.append('output:truncated')

    # 3. Missing answer tag
    if not pred_info.get('has_answer_tag') and status == 'success':
        errors.append('output:no_answer_tag')

    # 4. Answer extraction issues
    if pred_info.get('has_answer_tag') and pred_info.get('extracted_answer'):
        extracted = pred_info['extracted_answer']
        if len(extracted) > 500:
            errors.append('answer:too_long')
        if '\n' in extracted:
            errors.append('answer:multiline')

    # 5. Comparison with judge extraction
    comparison = compare_answers(pred_info.get('pred_answer'), row.get('res'))
    if comparison == 'mismatch' and pred_info.get('has_answer_tag'):
        errors.append('answer:judge_mismatch')

    # 6. Low reason score
    if reason_score < 0.3:
        errors.append('quality:very_low_rs')
    elif reason_score < 0.6:
        errors.append('quality:low_rs')

    # 7. Self-doubt / confusion
    if pred_info.get('doubt_count', 0) > 3:
        errors.append('behavior:self_doubt')

    if not errors:
        errors.append('other')

    return errors

# ── Process all samples ────────────────────────────────────────────────────
results = []
for idx, row in df.iterrows():
    pred = row.get('prediction')
    log = row.get('log')
    res = row.get('res')
    reason_score = row.get('reason_score', 0)
    task = row.get('task', 'unknown')
    gt_answer = row.get('answer', '')

    pred_info = analyze_prediction(pred)
    log_info = parse_log(log)

    # Determine if correct (using res column vs ground truth answer)
    # The 'res' column is the judge-extracted answer, we compare it with ground truth
    gt_str = str(gt_answer).strip().lower() if not pd.isna(gt_answer) else ''
    res_str = str(res).strip().lower() if not pd.isna(res) else ''
    is_correct = (gt_str == res_str) if gt_str and res_str else None

    errors = categorize_error(row, pred_info, log_info)

    results.append({
        'index': idx,
        'task': task,
        'status': log_info.get('status', 'unknown'),
        'pred_len': pred_info.get('pred_len', 0),
        'has_answer_tag': pred_info.get('has_answer_tag', False),
        'likely_truncated': pred_info.get('likely_truncated', False),
        'doubt_count': pred_info.get('doubt_count', 0),
        'extracted_answer': pred_info.get('extracted_answer'),
        'pred_answer': pred_info.get('pred_answer'),
        'res': str(res) if not pd.isna(res) else '',
        'gt_answer': str(gt_answer) if not pd.isna(gt_answer) else '',
        'reason_score': reason_score if not pd.isna(reason_score) else 0,
        'errors': errors,
        'is_correct': is_correct,
        'attempts': log_info.get('attempts', 1),
        'ppf_count': log_info.get('ppf_count', 0),
    })

rdf = pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("1. OVERALL STATUS DISTRIBUTION")
print("=" * 80)
status_counts = rdf['status'].value_counts()
for s, c in status_counts.items():
    pct = c / len(rdf) * 100
    print(f"  {s:<35s}: {c:4d} ({pct:5.1f}%)")
print()

print("=" * 80)
print("2. ACCURACY BY TASK TYPE")
print("=" * 80)
# Accuracy: res == gt_answer (judge-extracted answer matches ground truth)
correct_mask = rdf['is_correct'] == True
overall_acc = correct_mask.sum() / len(rdf) * 100
print(f"  Overall accuracy: {correct_mask.sum()}/{len(rdf)} = {overall_acc:.1f}%")
print()

task_acc = {}
for task in sorted(rdf['task'].unique()):
    subset = rdf[rdf['task'] == task]
    correct = (subset['is_correct'] == True).sum()
    task_acc[task] = correct / len(subset) * 100
    print(f"  {task:<40s}: {correct:3d}/{len(subset):3d} = {task_acc[task]:5.1f}%")

print()
print(f"  Mean reason_score by task:")
for task in sorted(rdf['task'].unique()):
    subset = rdf[rdf['task'] == task]
    mean_rs = subset['reason_score'].mean()
    print(f"  {task:<40s}: {mean_rs:.3f}")

print()
print("=" * 80)
print("3. ERROR MODE ANALYSIS")
print("=" * 80)

# Flatten all errors
all_errors = []
for _, row in rdf.iterrows():
    for e in row['errors']:
        all_errors.append({'task': row['task'], 'error': e, 'index': row['index'],
                           'reason_score': row['reason_score'], 'status': row['status']})
edf = pd.DataFrame(all_errors)

# Overall error distribution
print("\n3a. Error type distribution (all errors, including multiple per sample):")
error_counts = edf['error'].value_counts()
for e, c in error_counts.items():
    pct = c / len(rdf) * 100
    print(f"  {e:<40s}: {c:4d} ({pct:5.1f}% of samples)")

# Unique samples per error type
print("\n3b. Error type distribution (unique samples, count each sample once per error):")
error_sample_counts = edf.groupby('error')['index'].nunique().sort_values(ascending=False)
for e, c in error_sample_counts.items():
    pct = c / len(rdf) * 100
    print(f"  {e:<40s}: {c:4d} samples ({pct:5.1f}%)")

print()
print("=" * 80)
print("4. ERROR × TASK TYPE CROSS ANALYSIS")
print("=" * 80)

# For each error type, show task distribution
error_types_of_interest = [
    'pipeline:planning_parse_failed', 'pipeline:reasoning_failed',
    'output:truncated', 'output:no_answer_tag',
    'quality:very_low_rs', 'quality:low_rs',
    'behavior:self_doubt', 'answer:judge_mismatch'
]

for et in error_types_of_interest:
    subset = edf[edf['error'] == et]
    if len(subset) == 0:
        continue
    unique_samples = subset['index'].nunique()
    print(f"\n  [{et}] — {unique_samples} unique samples")
    task_dist = subset.groupby('task')['index'].nunique().sort_values(ascending=False)
    total_by_task = rdf.groupby('task')['index'].nunique()
    for task, count in task_dist.items():
        task_total = total_by_task.get(task, 1)
        pct_of_error = count / unique_samples * 100
        pct_of_task = count / task_total * 100
        print(f"    {task:<40s}: {count:3d} ({pct_of_error:5.1f}% of this error, {pct_of_task:5.1f}% of task's samples)")

print()
print("=" * 80)
print("5. TRUNCATION DEEP DIVE")
print("=" * 80)

truncated = rdf[rdf['likely_truncated'].apply(lambda x: x in [True, 'maybe'])]
print(f"  Total likely truncated: {len(truncated)}/{len(rdf)} ({len(truncated)/len(rdf)*100:.1f}%)")

# Truncation by task
print("\n  Truncation by task:")
for task in sorted(rdf['task'].unique()):
    task_total = len(rdf[rdf['task'] == task])
    task_trunc = len(rdf[(rdf['task'] == task) & (rdf['likely_truncated'].apply(lambda x: x in [True, 'maybe']))])
    pct = task_trunc / task_total * 100 if task_total > 0 else 0
    print(f"    {task:<40s}: {task_trunc:3d}/{task_total:3d} ({pct:5.1f}%)")

# Truncated vs non-truncated accuracy comparison
trunc_mask = rdf['likely_truncated'].apply(lambda x: x in [True, 'maybe'])
trunc_correct = (rdf[trunc_mask]['is_correct'] == True).sum()
non_trunc_correct = (rdf[~trunc_mask]['is_correct'] == True).sum()
print(f"\n  Accuracy comparison:")
print(f"    Truncated samples:    {trunc_correct}/{len(rdf[trunc_mask])} = {trunc_correct/max(len(rdf[trunc_mask]),1)*100:.1f}%")
print(f"    Non-truncated samples: {non_trunc_correct}/{len(rdf[~trunc_mask])} = {non_trunc_correct/max(len(rdf[~trunc_mask]),1)*100:.1f}%")

# Truncated reason_score
print(f"\n  Mean reason_score:")
print(f"    Truncated:      {rdf[trunc_mask]['reason_score'].mean():.3f}")
print(f"    Non-truncated:  {rdf[~trunc_mask]['reason_score'].mean():.3f}")

print()
print("=" * 80)
print("6. ANSWER TAG ANALYSIS")
print("=" * 80)

has_tag = rdf[rdf['has_answer_tag'] == True]
no_tag = rdf[rdf['has_answer_tag'] == False]
print(f"  Has <a> tag: {len(has_tag)}/{len(rdf)} ({len(has_tag)/len(rdf)*100:.1f}%)")
print(f"  No <a> tag:  {len(no_tag)}/{len(rdf)} ({len(no_tag)/len(rdf)*100:.1f}%)")

tag_correct = (has_tag['is_correct'] == True).sum()
no_tag_correct = (no_tag['is_correct'] == True).sum()
print(f"\n  Accuracy:")
print(f"    With <a> tag:    {tag_correct}/{len(has_tag)} = {tag_correct/max(len(has_tag),1)*100:.1f}%")
print(f"    Without <a> tag: {no_tag_correct}/{len(no_tag)} = {no_tag_correct/max(len(no_tag),1)*100:.1f}%")

print(f"\n  Mean reason_score:")
print(f"    With <a> tag:    {has_tag['reason_score'].mean():.3f}")
print(f"    Without <a> tag: {no_tag['reason_score'].mean():.3f}")

# No-tag status breakdown
print(f"\n  No <a> tag status breakdown:")
no_tag_status = no_tag.groupby('status')['index'].count().sort_values(ascending=False)
for s, c in no_tag_status.items():
    pct = c / len(no_tag) * 100
    print(f"    {s:<35s}: {c:3d} ({pct:5.1f}%)")

print()
print("=" * 80)
print("7. PIPELINE STAGE FAILURE ANALYSIS")
print("=" * 80)

for status in ['planning_parse_failed', 'planning_api_failed', 'execution_error',
               'reasoning_failed', 'planning_decompose_failed', 'success']:
    subset = rdf[rdf['status'] == status]
    if len(subset) == 0:
        continue
    print(f"\n  [{status}] — {len(subset)} samples")
    print(f"    Mean reason_score: {subset['reason_score'].mean():.3f}")
    correct = (subset['is_correct'] == True).sum()
    print(f"    Accuracy: {correct}/{len(subset)} = {correct/max(len(subset),1)*100:.1f}%")
    # Task distribution
    task_dist = subset.groupby('task')['index'].count().sort_values(ascending=False)
    for t, c in task_dist.items():
        print(f"      {t:<40s}: {c:3d}")

print()
print("=" * 80)
print("8. SELF-DOUBT / CONFUSION PATTERNS")
print("=" * 80)

doubt_mask = rdf['doubt_count'] > 3
doubtful = rdf[doubt_mask]
print(f"  Samples with high self-doubt (>3 patterns): {len(doubtful)}/{len(rdf)} ({len(doubtful)/len(rdf)*100:.1f}%)")
doubt_correct = (doubtful['is_correct'] == True).sum()
print(f"  Accuracy: {doubt_correct}/{len(doubtful)} = {doubt_correct/max(len(doubtful),1)*100:.1f}%")
print(f"  Mean reason_score: {doubtful['reason_score'].mean():.3f}")

# Doubt by task
print("\n  Self-doubt by task:")
for task in sorted(rdf['task'].unique()):
    task_total = len(rdf[rdf['task'] == task])
    task_doubt = len(rdf[(rdf['task'] == task) & doubt_mask])
    pct = task_doubt / task_total * 100 if task_total > 0 else 0
    print(f"    {task:<40s}: {task_doubt:3d}/{task_total:3d} ({pct:5.1f}%)")

print()
print("=" * 80)
print("9. PREDICTION LENGTH ANALYSIS")
print("=" * 80)

print(f"  Mean prediction length: {rdf['pred_len'].mean():.0f} chars")
print(f"  Median prediction length: {rdf['pred_len'].median():.0f} chars")
print(f"  Max prediction length: {rdf['pred_len'].max():.0f} chars")

# Length vs accuracy
rdf['len_bucket'] = pd.cut(rdf['pred_len'], bins=[0, 500, 1000, 2000, 5000, 10000, 100000],
                           labels=['<500', '500-1K', '1K-2K', '2K-5K', '5K-10K', '>10K'])
print("\n  Accuracy by prediction length:")
for bucket in ['<500', '500-1K', '1K-2K', '2K-5K', '5K-10K', '>10K']:
    subset = rdf[rdf['len_bucket'] == bucket]
    if len(subset) == 0:
        continue
    correct = (subset['is_correct'] == True).sum()
    rs_mean = subset['reason_score'].mean()
    print(f"    {bucket:<10s}: {correct:3d}/{len(subset):3d} acc={correct/max(len(subset),1)*100:5.1f}%  mean_RS={rs_mean:.3f}")

# Length by status
print("\n  Mean prediction length by status:")
for status in sorted(rdf['status'].unique()):
    subset = rdf[rdf['status'] == status]
    if len(subset) == 0:
        continue
    print(f"    {status:<35s}: {subset['pred_len'].mean():8.0f} chars (n={len(subset)})")

print()
print("=" * 80)
print("10. REASON_SCORE DISTRIBUTION ANALYSIS")
print("=" * 80)

rs_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
rs_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
rdf['rs_bucket'] = pd.cut(rdf['reason_score'], bins=rs_bins, labels=rs_labels, include_lowest=True)

print("  Reason score distribution:")
for bucket in rs_labels:
    subset = rdf[rdf['rs_bucket'] == bucket]
    if len(subset) == 0:
        continue
    correct = (subset['is_correct'] == True).sum()
    print(f"    RS {bucket:<10s}: {len(subset):3d} samples ({len(subset)/len(rdf)*100:5.1f}%), accuracy={correct/max(len(subset),1)*100:5.1f}%")

print()
print("=" * 80)
print("11. ERROR CO-OCCURRENCE MATRIX")
print("=" * 80)

# Which errors tend to appear together?
error_types_list = ['pipeline:planning_parse_failed', 'pipeline:reasoning_failed',
                    'output:truncated', 'output:no_answer_tag',
                    'quality:very_low_rs', 'quality:low_rs',
                    'behavior:self_doubt', 'answer:judge_mismatch',
                    'answer:too_long', 'answer:multiline']

# Build co-occurrence
print("\n  Error co-occurrence (top pairs):")
co_occur = Counter()
for _, row in rdf.iterrows():
    sample_errors = set(row['errors'])
    for e1 in sample_errors:
        for e2 in sample_errors:
            if e1 < e2:
                co_occur[(e1, e2)] += 1

for (e1, e2), count in co_occur.most_common(25):
    pct = count / len(rdf) * 100
    print(f"    {e1} + {e2}: {count:3d} ({pct:4.1f}%)")

print()
print("=" * 80)
print("12. MULTI-ERROR SAMPLES (compound failures)")
print("=" * 80)

rdf['n_errors'] = rdf['errors'].apply(len)
error_count_dist = rdf['n_errors'].value_counts().sort_index()
print("  Number of error types per sample:")
for n, c in error_count_dist.items():
    pct = c / len(rdf) * 100
    subset = rdf[rdf['n_errors'] == n]
    correct = (subset['is_correct'] == True).sum()
    print(f"    {n} error(s): {c:3d} samples ({pct:5.1f}%), accuracy={correct/max(len(subset),1)*100:5.1f}%")

print()
print("=" * 80)
print("13. ANSWER QUALITY: PRED vs RES vs GT")
print("=" * 80)

# For samples with <a> tag, compare extracted answer with judge answer and GT
tagged = rdf[(rdf['has_answer_tag']) & (rdf['extracted_answer'].notna())]
print(f"  Samples with extractable answer: {len(tagged)}")

# Judge extraction match rate
match_counts = Counter()
for _, row in tagged.iterrows():
    comp = compare_answers(row.get('pred_answer'), row.get('res'))
    match_counts[comp] += 1

print("\n  Predicted answer vs Judge-extracted answer (res):")
for comp, c in match_counts.items():
    pct = c / len(tagged) * 100
    print(f"    {comp:<25s}: {c:3d} ({pct:5.1f}%)")

# For mismatches, what happened?
mismatches = tagged[[compare_answers(row.get('pred_answer'), row.get('res')) == 'mismatch'
                     for _, row in tagged.iterrows()]]
print(f"\n  Answer mismatches between model output and judge extraction: {len(mismatches)}")
if len(mismatches) > 0:
    print(f"    Mean reason_score of mismatches: {mismatches['reason_score'].mean():.3f}")
    mismatch_tasks = mismatches.groupby('task')['index'].count().sort_values(ascending=False)
    for t, c in mismatch_tasks.items():
        print(f"      {t}: {c}")

print()
print("=" * 80)
print("14. TASK-SPECIFIC ERROR PROFILES")
print("=" * 80)

# For each task, show the top error types
for task in sorted(rdf['task'].unique()):
    task_subset = edf[edf['task'] == task]
    if len(task_subset) == 0:
        continue
    task_n = len(rdf[rdf['task'] == task])
    task_correct = (rdf[rdf['task'] == task]['is_correct'] == True).sum()
    print(f"\n  [{task}] — {task_n} samples, accuracy={task_correct/task_n*100:.1f}%")
    error_dist = task_subset.groupby('error')['index'].nunique().sort_values(ascending=False)
    for e, c in error_dist.head(8).items():
        pct = c / task_n * 100
        print(f"    {e:<40s}: {c:3d}/{task_n} ({pct:5.1f}%)")

print()
print("=" * 80)
print("15. SUCCESS CASE CHARACTERISTICS")
print("=" * 80)

# Compare correct vs incorrect samples on key metrics
correct_set = rdf[rdf['is_correct'] == True]
incorrect_set = rdf[rdf['is_correct'] == False]
unknown_set = rdf[rdf['is_correct'].isna()]

print(f"  Correct: {len(correct_set)}, Incorrect: {len(incorrect_set)}, Unknown: {len(unknown_set)}")
print(f"\n  Metric comparison (correct vs incorrect):")
print(f"    Mean pred_len:      {correct_set['pred_len'].mean():.0f} vs {incorrect_set['pred_len'].mean():.0f}")
print(f"    Mean reason_score:   {correct_set['reason_score'].mean():.3f} vs {incorrect_set['reason_score'].mean():.3f}")
print(f"    Mean doubt_count:    {correct_set['doubt_count'].mean():.1f} vs {incorrect_set['doubt_count'].mean():.1f}")
print(f"    Has <a> tag rate:    {correct_set['has_answer_tag'].mean()*100:.0f}% vs {incorrect_set['has_answer_tag'].mean()*100:.0f}%")

print()
print("=" * 80)
print("16. KEY INSIGHTS SUMMARY")
print("=" * 80)

total = len(rdf)
# Major bottleneck percentages
ppf_count = len(rdf[rdf['status'] == 'planning_parse_failed'])
trunc_count = len(rdf[rdf['likely_truncated'].apply(lambda x: x in [True, 'maybe'])])
no_tag_count = len(rdf[(rdf['has_answer_tag'] == False)])
rs_low_count = len(rdf[rdf['reason_score'] < 0.4])
doubt_count = len(rdf[rdf['doubt_count'] > 3])

print(f"""
  Pipeline Status:
    ┌─ Planning parse failed:  {ppf_count:3d}/{total} ({ppf_count/total*100:4.1f}%) ← #1 pipeline blocker
    ├─ Reasoning failed:       {len(rdf[rdf['status']=='reasoning_failed']):3d}/{total} ({len(rdf[rdf['status']=='reasoning_failed'])/total*100:4.1f}%)
    ├─ Execution error:        {len(rdf[rdf['status']=='execution_error']):3d}/{total}
    └─ Success (all stages):   {len(rdf[rdf['status']=='success']):3d}/{total}

  Output Quality:
    ├─ Truncated outputs:      {trunc_count:3d}/{total} ({trunc_count/total*100:4.1f}%)
    ├─ No <a> tag:             {no_tag_count:3d}/{total} ({no_tag_count/total*100:4.1f}%)
    ├─ Very low RS (<0.4):     {rs_low_count:3d}/{total} ({rs_low_count/total*100:4.1f}%)
    └─ High self-doubt:        {doubt_count:3d}/{total} ({doubt_count/total*100:4.1f}%)

  Task Weaknesses:
    ├─ Spatial:       {task_acc.get('spatial reasoning', 0):.1f}% acc, {rdf[rdf['task']=='spatial reasoning']['reason_score'].mean():.3f} RS
    ├─ Mathematical:  {task_acc.get('mathematical reasoning', 0):.1f}% acc, {rdf[rdf['task']=='mathematical reasoning']['reason_score'].mean():.3f} RS
    ├─ Multidisciplinary: {task_acc.get('multidisciplinary knowledge reasoning', 0):.1f}% acc, {rdf[rdf['task']=='multidisciplinary knowledge reasoning']['reason_score'].mean():.3f} RS
    └─ Logical:       {task_acc.get('logical reasoning', 0):.1f}% acc, {rdf[rdf['task']=='logical reasoning']['reason_score'].mean():.3f} RS
""")

print("=" * 80)
print("Analysis complete.")
