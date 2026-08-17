#!/usr/bin/env python3
"""Supplemental analysis: extraction error patterns, false positives/negatives."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import re
import numpy as np
from collections import Counter

XLSX = r"T:\PyCharmProject\OCR-Reasoning\OCR-Reasoning-main\outputs\Qwen3.5-9B-OCRAgentV5\T20260730_G533ef742\Qwen3.5-9B-OCRAgentV5_OCR_Reasoning_qwen-plus.xlsx"
df = pd.read_excel(XLSX)

def normalize(s):
    if pd.isna(s): return ''
    return re.sub(r'\s+', '', str(s))

df['res_norm'] = df['res'].apply(normalize)
df['ans_norm'] = df['answer'].apply(normalize)
df['is_correct'] = (df['res_norm'] == df['ans_norm']) & (df['res_norm'] != '')

# ── 1. Extraction Error Deep Dive ──
print("=" * 80)
print("1. EXTRACTION ERROR PATTERNS (samples with <a> tag but wrong answer)")
print("=" * 80)

wrong_tagged = []
for idx, row in df.iterrows():
    pred = str(row['prediction']) if pd.notna(row['prediction']) else ''
    has_a = bool(re.search(r'<a>.*?</a>', pred, re.DOTALL))
    if has_a and not row['is_correct']:
        a_m = re.search(r'<a>(.*?)</a>', pred, re.DOTALL)
        a_content = a_m.group(1).strip() if a_m else ''
        wrong_tagged.append({
            'index': idx,
            'task': row['task'],
            'model_answer': a_content[:120],
            'res': str(row['res'])[:120] if pd.notna(row['res']) else '',
            'gt': str(row['answer'])[:120] if pd.notna(row['answer']) else '',
            'rs': row['reason_score'],
            'pred_len': len(pred),
            'pred_start': pred[:200],
        })

print(f"Total wrong with <a> tag: {len(wrong_tagged)}")
print()

# Categorize the error types
for e in wrong_tagged:
    a = normalize(e['model_answer'])
    g = normalize(e['gt'])
    r = normalize(e['res'])

    # Classify error type
    if not a:
        e['error_type'] = 'empty_answer'
    elif a == r and a != g:
        e['error_type'] = 'judge_and_model_agree_wrong'
    elif a != r and r == g:
        e['error_type'] = 'judge_correct_model_wrong'
    elif a == g and r != g:
        e['error_type'] = 'model_correct_judge_wrong'
    else:
        e['error_type'] = 'both_wrong_different'

    # Numeric comparison
    try:
        a_num = float(re.sub(r'[^\d.\-]', '', a))
        g_num = float(re.sub(r'[^\d.\-]', '', g))
        if a_num != 0:
            e['relative_error'] = abs(a_num - g_num) / abs(g_num)
        else:
            e['relative_error'] = 999
    except:
        e['relative_error'] = None

error_types = Counter(e['error_type'] for e in wrong_tagged)
print("Error sub-types:")
for et, c in error_types.items():
    print(f"  {et}: {c}")

print("\nDetailed examples:")
for e in wrong_tagged:
    print(f"\n  [{e['index']}] {e['task']} | RS={e['rs']} | {e['error_type']} | rel_err={e['relative_error']}")
    print(f"    Model: {e['model_answer'][:100]}")
    print(f"    Judge: {e['res'][:100]}")
    print(f"    GT:    {e['gt'][:100]}")
    if e['error_type'] == 'judge_and_model_agree_wrong':
        print(f"    *** BOTH model and judge agree on WRONG answer ***")

# ── 2. "Correct but no tag" samples ──
print("\n" + "=" * 80)
print("2. CORRECT SAMPLES WITHOUT <a> TAG (anomalies)")
print("=" * 80)

for idx, row in df.iterrows():
    pred = str(row['prediction']) if pd.notna(row['prediction']) else ''
    has_a = bool(re.search(r'<a>.*?</a>', pred, re.DOTALL))
    if not has_a and row['is_correct'] and len(pred) > 10:
        print(f"\n  [{idx}] {row['task']} | RS={row['reason_score']} | pred_len={len(pred)}")
        print(f"    Pred: {pred[:300]}")
        print(f"    Res:  {row['res']}")

# ── 3. High RS but wrong answer ──
print("\n" + "=" * 80)
print("3. HIGH RS (>0.7) BUT WRONG ANSWER (good reasoning, wrong result)")
print("=" * 80)

count = 0
for idx, row in df.iterrows():
    if not row['is_correct'] and row['reason_score'] >= 0.7:
        count += 1
        pred = str(row['prediction'])[:400] if pd.notna(row['prediction']) else 'NaN'
        print(f"\n  [{idx}] {row['task']} | RS={row['reason_score']}")
        print(f"    Res:  {row['res']}")
        print(f"    GT:   {row['answer']}")
        print(f"    Pred: {pred[:300]}")
print(f"\n  Total: {count} samples with RS>=0.7 but wrong answer")

# ── 4. Low RS but correct answer ──
print("\n" + "=" * 80)
print("4. LOW RS (<0.4) BUT CORRECT ANSWER (lucky guess?)")
print("=" * 80)

count = 0
for idx, row in df.iterrows():
    if row['is_correct'] and row['reason_score'] < 0.4:
        count += 1
        pred = str(row['prediction'])[:400] if pd.notna(row['prediction']) else 'NaN'
        print(f"\n  [{idx}] {row['task']} | RS={row['reason_score']}")
        print(f"    Res:  {row['res']}")
        print(f"    GT:   {row['answer']}")
        print(f"    Pred: {pred[:300]}")
print(f"\n  Total: {count} samples with RS<0.4 but correct answer")

# ── 5. Truncation content analysis ──
print("\n" + "=" * 80)
print("5. TRUNCATION PATTERN ANALYSIS")
print("=" * 80)

truncated_samples = []
for idx, row in df.iterrows():
    pred = str(row['prediction']) if pd.notna(row['prediction']) else ''
    has_a = bool(re.search(r'<a>.*?</a>', pred, re.DOTALL))
    if not has_a and len(pred) > 5000:
        # Check what the model is doing instead of answering
        last_500 = pred[-500:]
        # Find common patterns
        wait_count = len(re.findall(r'\b[Ww]ait\b', pred))
        check_count = len(re.findall(r'\bcheck\b', pred, re.IGNORECASE))
        need_count = len(re.findall(r'\bneed to\b', pred, re.IGNORECASE))
        step_count = len(re.findall(r'Step\s*\d', pred, re.IGNORECASE))
        truncated_samples.append({
            'index': idx,
            'task': row['task'],
            'len': len(pred),
            'wait': wait_count,
            'check': check_count,
            'need': need_count,
            'step': step_count,
            'rs': row['reason_score'],
            'last_200': pred[-200:],
        })

print(f"Truncated samples (>5K, no <a>): {len(truncated_samples)}")
if truncated_samples:
    avg_wait = np.mean([s['wait'] for s in truncated_samples])
    avg_check = np.mean([s['check'] for s in truncated_samples])
    avg_need = np.mean([s['need'] for s in truncated_samples])
    avg_step = np.mean([s['step'] for s in truncated_samples])
    print(f"  Avg 'Wait' mentions: {avg_wait:.0f}")
    print(f"  Avg 'check' mentions: {avg_check:.0f}")
    print(f"  Avg 'need to' mentions: {avg_need:.0f}")
    print(f"  Avg 'Step N' mentions: {avg_step:.0f}")

    print("\n  Last 200 chars of each truncated sample:")
    for s in truncated_samples:
        print(f"\n  [{s['index']}] {s['task']} | len={s['len']} | RS={s['rs']} | wait={s['wait']} check={s['check']}")
        print(f"    ...{s['last_200'][:200]}")

# ── 6. Accuracy loss attribution (precise) ──
print("\n" + "=" * 80)
print("6. PRECISE ACCURACY LOSS ATTRIBUTION")
print("=" * 80)

N = len(df)
overall_acc = df['is_correct'].sum() / N * 100
max_acc = 100

# Loss from API failures
api_fail_mask = df['log'].apply(lambda x: 'Failed to obtain answer via API' in str(x) if pd.notna(x) else False)
api_fail_n = api_fail_mask.sum()
api_loss = api_fail_n / N * 100  # they're all wrong

# Loss from truncation (among non-API-failures)
non_api = df[~api_fail_mask]
trunc_mask = non_api['prediction'].apply(
    lambda x: not bool(re.search(r'<a>.*?</a>', str(x), re.DOTALL)) and len(str(x)) > 5000
    if pd.notna(x) else False
)
trunc_n = trunc_mask.sum()
trunc_correct_n = non_api[trunc_mask]['is_correct'].sum()
trunc_loss = (trunc_n - trunc_correct_n) / N * 100

# Loss from clean extraction errors (has <a> tag but wrong)
clean_wrong_mask = non_api['prediction'].apply(
    lambda x: bool(re.search(r'<a>.*?</a>', str(x), re.DOTALL))
    if pd.notna(x) else False
)
clean_wrong = non_api[clean_wrong_mask & (~non_api['is_correct'])]
clean_loss = len(clean_wrong) / N * 100

# Residual: wrong with no <a> and short output, or other
other_wrong = non_api[(~non_api['is_correct']) & (~trunc_mask) & (~clean_wrong_mask)]
other_loss = len(other_wrong) / N * 100

print(f"""
  Maximum possible accuracy: 100.0%
  Actual accuracy:           {overall_acc:.1f}%
  Total accuracy gap:        {max_acc - overall_acc:.1f}%

  Loss attribution:
    API failures:            -{api_loss:.1f}%  ({api_fail_n} samples, all wrong)
    Output truncation:       -{trunc_loss:.1f}%  ({trunc_n} truncated, {trunc_correct_n} correct despite truncation)
    Clean extraction errors: -{clean_loss:.1f}%  ({len(clean_wrong)} samples, has <a> but wrong answer)
    Other losses:            -{other_loss:.1f}%  ({len(other_wrong)} samples)
    ─────────────────────
    Explained loss:           {api_loss + trunc_loss + clean_loss + other_loss:.1f}%
""")

print("=" * 80)
print("Analysis Complete.")
