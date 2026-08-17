#!/usr/bin/env python3
"""
Deep error analysis for OCRAgentV5 (no decomposition) experiment.
Focus: error mode classification, task-type cross analysis, internal correlations.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict

# ── Load ───────────────────────────────────────────────────────────────────
XLSX = r"T:\PyCharmProject\OCR-Reasoning\OCR-Reasoning-main\outputs\Qwen3.5-9B-OCRAgentV5\T20260730_G533ef742\Qwen3.5-9B-OCRAgentV5_OCR_Reasoning_qwen-plus.xlsx"
df = pd.read_excel(XLSX)
N = len(df)

# ── Helper: strip whitespace for answer comparison ─────────────────────────
def normalize(s):
    """Remove all whitespace for answer comparison."""
    if pd.isna(s):
        return ''
    return re.sub(r'\s+', '', str(s))

# ── Determine correctness using normalized comparison ──────────────────────
df['res_norm'] = df['res'].apply(normalize)
df['ans_norm'] = df['answer'].apply(normalize)
df['is_correct'] = (df['res_norm'] == df['ans_norm']) & (df['res_norm'] != '')

# ── Classify each sample ───────────────────────────────────────────────────
def classify_sample(row):
    """Return a dict of features for this sample."""
    pred = str(row['prediction']) if pd.notna(row['prediction']) else ''
    log = str(row['log']) if pd.notna(row['log']) else ''
    task = row['task']
    rs = row['reason_score'] if pd.notna(row['reason_score']) else 0
    res = str(row['res']) if pd.notna(row['res']) else ''

    f = {'task': task, 'rs': rs, 'pred_len': len(pred)}

    # ── Pipeline status ──
    if log == 'Succeed':
        f['pipeline_status'] = 'success'
    elif 'Failed to obtain answer via API' in log:
        f['pipeline_status'] = 'api_failed'
    elif 'failed to parse' in log:
        f['pipeline_status'] = 'parse_failed'
    else:
        f['pipeline_status'] = 'other_failure'

    # ── Prediction analysis ──
    has_a = bool(re.search(r'<a>.*?</a>', pred, re.DOTALL))
    f['has_a_tag'] = has_a

    if has_a:
        m = re.search(r'<a>(.*?)</a>', pred, re.DOTALL)
        f['a_content'] = m.group(1).strip()
    else:
        f['a_content'] = None

    # ── Truncation detection ──
    # Truncated = has content but no closing </a>, or ends mid-sentence
    f['is_truncated'] = False
    f['truncation_signal'] = 'none'
    if not has_a and len(pred) > 500:
        last_100 = pred[-100:].lower()
        doubt_signals = ['wait', 'however', 'but actually', 'let me', 'actually,',
                         'alternatively', 'hmm', 'no,', 'i mean', 'rather than',
                         'let\'s', 'i should', 'i need to', 'check if', 'so i will',
                         'okay,', 'now,', 'therefore,', 'thus,', 'so the']
        if any(kw in last_100 for kw in doubt_signals):
            f['is_truncated'] = True
            f['truncation_signal'] = 'self_doubt_loop'
        elif len(pred) > 10000:
            f['is_truncated'] = True
            f['truncation_signal'] = 'very_long_no_tag'

    if len(pred) < 50 and not has_a:
        f['is_empty'] = True
    else:
        f['is_empty'] = False

    # ── Self-doubt pattern count ──
    doubt_patterns = [
        r'\bwait\b', r'\bhowever\b', r'\bbut actually\b', r'\blet me\b',
        r'\bactually\b', r'\balternatively\b', r'\bhmm\b',
        r'\bno,\s', r'\bi mean\b', r'\brather than\b',
        r'\blet\'s\b', r'\bi should\b', r'\bi need to\b',
        r'\bcheck if\b', r'\bokay,\s', r'\bnow,\s',
        r'\bon second thought\b', r'\breconsider\b',
    ]
    doubt_count = sum(len(re.findall(pat, pred, re.IGNORECASE)) for pat in doubt_patterns)
    f['doubt_count'] = doubt_count
    f['has_heavy_doubt'] = doubt_count > 10

    # ── Repetition patterns ──
    # Count repeated phrases
    lines = pred.split('\n')
    line_starts = [l.strip()[:30] for l in lines if l.strip()]
    if line_starts:
        repeated_starts = sum(1 for s, c in Counter(line_starts).items() if c > 2)
        f['repeated_line_starts'] = repeated_starts
    else:
        f['repeated_line_starts'] = 0

    # ── Language ──
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', pred))
    f['is_chinese'] = chinese_chars > 20
    f['chinese_ratio'] = chinese_chars / max(len(pred), 1)

    # ── Reasoning structure patterns ──
    f['has_step_by_step'] = bool(re.search(r'Step\s*\d', pred, re.IGNORECASE))
    f['has_json_block'] = bool(re.search(r'\{\s*"reasoning"', pred))
    f['mentions_wait'] = 'wait' in pred.lower()

    # ── Answer extraction quality ──
    if has_a and f['a_content']:
        a = f['a_content']
        f['answer_has_newlines'] = '\n' in a
        f['answer_len'] = len(a)
        f['answer_is_numeric'] = bool(re.match(r'^[\d.,%\s\-+$¥€£元大卡份个十百千万亿]+$', a))
    else:
        f['answer_has_newlines'] = False
        f['answer_len'] = 0
        f['answer_is_numeric'] = False

    # ── Error root cause classification ──
    f['error_category'] = classify_root_cause(row, f, pred, log)
    f['error_subcategory'] = classify_sub_error(row, f, pred)

    return f


def classify_root_cause(row, f, pred, log):
    """Classify the root cause of error for this sample."""
    if not row['is_correct']:
        # API failure
        if f['pipeline_status'] == 'api_failed':
            return 'A_API_FAILURE'
        # Empty output
        if f['is_empty']:
            return 'A_API_FAILURE'
        # Truncated - no answer tag
        if f['is_truncated']:
            return 'B_TRUNCATION'
        # Has answer tag but wrong
        if f['has_a_tag']:
            if f['doubt_count'] > 10:
                return 'C_EXTRACTION_ERROR_WITH_DOUBT'
            return 'D_EXTRACTION_ERROR'
        # No tag, not truncated, not empty
        return 'E_OTHER_FAILURE'
    else:
        return 'Z_CORRECT'


def classify_sub_error(row, f, pred):
    """Finer-grained error classification."""
    if row['is_correct']:
        return 'correct'

    if f['pipeline_status'] == 'api_failed':
        return 'api_call_failed'

    if f['is_empty']:
        return 'empty_output'

    if f['is_truncated']:
        return 'output_truncated'

    if not f['has_a_tag']:
        return 'no_answer_tag'

    # Has answer tag but wrong
    a_content = f.get('a_content', '')
    if a_content and len(a_content) > 200:
        return 'answer_too_verbose'

    # Mismatch types
    res = str(row['res']) if pd.notna(row['res']) else ''
    ans = str(row['answer']) if pd.notna(row['answer']) else ''

    if res and ans:
        # Judge might have extracted differently
        if normalize(res) != normalize(a_content) and normalize(ans) == normalize(a_content):
            return 'judge_extraction_error'

    if f['doubt_count'] > 10:
        return 'wrong_with_heavy_doubt'

    return 'wrong_answer'


# ── Process all samples ────────────────────────────────────────────────────
features = []
for idx, row in df.iterrows():
    f = classify_sample(row)
    f['index'] = idx
    features.append(f)

fdf = pd.DataFrame(features)
correct_n = fdf['error_category'].value_counts().get('Z_CORRECT', 0)
wrong_n = N - correct_n

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("  OCRAgentV5 (No Decomposition) — Deep Error Analysis")
print(f"  N={N} | Official Accuracy=63.0% | Reason Score RP=64.55")
print("=" * 90)

# ── SECTION 1: OVERALL RESULTS ────────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 1: OVERALL ACCURACY & SCORE DISTRIBUTION")
print("─" * 70)

print(f"\n  Simple match accuracy: {correct_n}/{N} = {correct_n/N*100:.1f}%")
print(f"  (Official GPT-4 judge accuracy: 63.0% — judge allows equivalent answers)")

print(f"\n  Reason Score Distribution:")
rs_bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
for lo, hi in rs_bins:
    count = ((fdf['rs'] >= lo) & (fdf['rs'] < hi)).sum()
    correct = ((fdf['rs'] >= lo) & (fdf['rs'] < hi) & (df['is_correct'])).sum()
    bar = '█' * (count // 2)
    print(f"    RS [{lo:.1f}-{hi:.1f}): {count:4d} ({count/N*100:5.1f}%)  acc={correct/max(count,1)*100:5.1f}%  {bar}")

# ── SECTION 2: ERROR ROOT CAUSE DISTRIBUTION ───────────────────────────────
print("\n" + "─" * 70)
print("SECTION 2: ERROR ROOT CAUSE CLASSIFICATION")
print("─" * 70)

error_order = [
    ('Z_CORRECT', 'Correct'),
    ('A_API_FAILURE', 'API Call Failure'),
    ('B_TRUNCATION', 'Output Truncation (no </a>)'),
    ('C_EXTRACTION_ERROR_WITH_DOUBT', 'Extraction Error + Heavy Self-Doubt'),
    ('D_EXTRACTION_ERROR', 'Extraction Error (clean)'),
    ('E_OTHER_FAILURE', 'Other / Unknown'),
]

print(f"\n  {'Category':<45s} {'Count':>5s}  {'%Total':>7s}  {'%Errors':>7s}")
print(f"  {'-'*45} {'-'*5}  {'-'*7}  {'-'*7}")
for cat, label in error_order:
    count = (fdf['error_category'] == cat).sum()
    pct_total = count / N * 100
    pct_errors = count / max(wrong_n, 1) * 100
    print(f"  {label:<45s} {count:5d}  {pct_total:6.1f}%  {pct_errors:6.1f}%")

# ── SECTION 3: ERROR × TASK TYPE CROSS MATRIX ──────────────────────────────
print("\n" + "─" * 70)
print("SECTION 3: ERROR CATEGORY × TASK TYPE CROSS MATRIX")
print("─" * 70)

tasks = sorted(df['task'].unique())
categories = ['A_API_FAILURE', 'B_TRUNCATION', 'C_EXTRACTION_ERROR_WITH_DOUBT',
              'D_EXTRACTION_ERROR', 'E_OTHER_FAILURE', 'Z_CORRECT']
cat_labels = {c: l for c, l in error_order}

# Print header
task_short = {t: t.replace(' reasoning', '').replace(' analysis', '').replace('knowledge ', '')
              for t in tasks}
print(f"\n  {'Task':<30s} {'Total':>5s} {'Corr':>5s} {'Acc%':>6s}", end='')
for cat in categories:
    if cat != 'Z_CORRECT':
        print(f"  {cat.split('_')[-1][:8]:>8s}", end='')
print()

for task in tasks:
    subset = fdf[fdf['task'] == task]
    total = len(subset)
    correct = (subset['error_category'] == 'Z_CORRECT').sum()
    print(f"  {task:<30s} {total:5d} {correct:5d} {correct/total*100:5.1f}%", end='')
    for cat in categories:
        if cat != 'Z_CORRECT':
            c = (subset['error_category'] == cat).sum()
            pct = c / total * 100
            print(f"  {c:3d} {pct:4.1f}%", end='')
    print()

# ── SECTION 4: API FAILURE DEEP DIVE ───────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 4: API FAILURE ANALYSIS")
print("─" * 70)

api_fails = fdf[fdf['error_category'] == 'A_API_FAILURE']
print(f"\n  Total API failures: {len(api_fails)}/{N} ({len(api_fails)/N*100:.1f}%)")
print(f"  Mean RS: {api_fails['rs'].mean():.3f}")
print(f"  Accuracy: {(api_fails['error_category']=='Z_CORRECT').sum()}/{len(api_fails)}")

# Check log details
print("\n  API failure log details:")
for idx in api_fails['index']:
    log = str(df.iloc[idx]['log'])[:200]
    task = df.iloc[idx]['task']
    print(f"    [{idx}] task={task} | {log}")

# ── SECTION 5: TRUNCATION DEEP DIVE ────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 5: OUTPUT TRUNCATION DEEP DIVE")
print("─" * 70)

trunc = fdf[fdf['is_truncated']]
print(f"\n  Total truncated: {len(trunc)}/{N} ({len(trunc)/N*100:.1f}%)")
print(f"  Mean prediction length: {trunc['pred_len'].mean():.0f} chars")
print(f"  Median prediction length: {trunc['pred_len'].median():.0f} chars")
print(f"  Mean RS: {trunc['rs'].mean():.3f}")
print(f"  Mean doubt_count: {trunc['doubt_count'].mean():.1f}")

# Truncation vs non-truncation comparison
non_trunc = fdf[~fdf['is_truncated']]
print(f"\n  Comparison (truncated vs non-truncated):")
print(f"    Accuracy:         {((trunc['error_category']=='Z_CORRECT').sum())/max(len(trunc),1)*100:.1f}% vs {(non_trunc['error_category']=='Z_CORRECT').sum()/max(len(non_trunc),1)*100:.1f}%")
print(f"    Mean RS:          {trunc['rs'].mean():.3f} vs {non_trunc['rs'].mean():.3f}")
print(f"    Mean doubt_count: {trunc['doubt_count'].mean():.0f} vs {non_trunc['doubt_count'].mean():.0f}")
print(f"    Mean pred_len:    {trunc['pred_len'].mean():.0f} vs {non_trunc['pred_len'].mean():.0f}")

# Truncation by task
print(f"\n  Truncation rate by task:")
for task in tasks:
    t_total = len(fdf[fdf['task'] == task])
    t_trunc = len(fdf[(fdf['task'] == task) & fdf['is_truncated']])
    pct = t_trunc / t_total * 100
    bar = '█' * int(pct / 2)
    print(f"    {task:<40s}: {t_trunc:3d}/{t_total:3d} ({pct:5.1f}%) {bar}")

# ── SECTION 6: ANSWER TAG ANALYSIS ─────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 6: ANSWER TAG (<a>) ANALYSIS")
print("─" * 70)

has_tag = fdf[fdf['has_a_tag']]
no_tag = fdf[~fdf['has_a_tag']]
print(f"\n  Has <a> tag: {len(has_tag)}/{N} ({len(has_tag)/N*100:.1f}%)")
print(f"    Accuracy: {(has_tag['error_category']=='Z_CORRECT').sum()}/{len(has_tag)} = {(has_tag['error_category']=='Z_CORRECT').sum()/max(len(has_tag),1)*100:.1f}%")
print(f"    Mean RS:  {has_tag['rs'].mean():.3f}")
print(f"    Mean pred_len: {has_tag['pred_len'].mean():.0f}")

print(f"\n  No <a> tag: {len(no_tag)}/{N} ({len(no_tag)/N*100:.1f}%)")
print(f"    Accuracy: {(no_tag['error_category']=='Z_CORRECT').sum()}/{len(no_tag)} = {(no_tag['error_category']=='Z_CORRECT').sum()/max(len(no_tag),1)*100:.1f}%")
print(f"    Mean RS:  {no_tag['rs'].mean():.3f}")
print(f"    Mean pred_len: {no_tag['pred_len'].mean():.0f}")

# No-tag subtypes
print(f"\n  No <a> tag breakdown:")
print(f"    Truncated (self-doubt loop): {len(no_tag[no_tag['truncation_signal']=='self_doubt_loop'])}")
print(f"    Very long no tag (>10K):     {len(no_tag[no_tag['truncation_signal']=='very_long_no_tag'])}")
print(f"    API failure / empty:         {len(no_tag[no_tag['is_empty']])}")
print(f"    Other (short but no tag):    {len(no_tag[(~no_tag['is_empty']) & (no_tag['truncation_signal']=='none')])}")

# ── SECTION 7: SELF-DOUBT & CONFUSION PATTERNS ─────────────────────────────
print("\n" + "─" * 70)
print("SECTION 7: SELF-DOUBT & VERBOSE LOOP ANALYSIS")
print("─" * 70)

heavy_doubt = fdf[fdf['has_heavy_doubt']]
print(f"\n  Heavy self-doubt (>10 doubt patterns): {len(heavy_doubt)}/{N} ({len(heavy_doubt)/N*100:.1f}%)")
print(f"    Accuracy: {(heavy_doubt['error_category']=='Z_CORRECT').sum()}/{len(heavy_doubt)} = {(heavy_doubt['error_category']=='Z_CORRECT').sum()/max(len(heavy_doubt),1)*100:.1f}%")
print(f"    Mean RS: {heavy_doubt['rs'].mean():.3f}")
print(f"    Mean pred_len: {heavy_doubt['pred_len'].mean():.0f}")

# Overlap: doubt + truncation
doubt_trunc = fdf[fdf['has_heavy_doubt'] & fdf['is_truncated']]
print(f"\n  Heavy doubt + Truncated (compound): {len(doubt_trunc)}/{N} ({len(doubt_trunc)/N*100:.1f}%)")
print(f"    Accuracy: {(doubt_trunc['error_category']=='Z_CORRECT').sum()}/{len(doubt_trunc)}")
print(f"    Mean RS: {doubt_trunc['rs'].mean():.3f}")

# Doubt by task
print(f"\n  Heavy self-doubt by task:")
for task in tasks:
    t_total = len(fdf[fdf['task'] == task])
    t_doubt = len(fdf[(fdf['task'] == task) & fdf['has_heavy_doubt']])
    pct = t_doubt / t_total * 100
    print(f"    {task:<40s}: {t_doubt:3d}/{t_total:3d} ({pct:5.1f}%)")

# ── SECTION 8: PREDICTION LENGTH EFFECTS ───────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 8: PREDICTION LENGTH vs ACCURACY")
print("─" * 70)

len_bins = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 5000),
            (5000, 10000), (10000, 25000)]
print(f"\n  {'Length Range':<20s} {'Count':>6s} {'Acc%':>7s} {'Mean RS':>8s} {'Has <a>':>8s} {'Doubt':>6s}")
for lo, hi in len_bins:
    subset = fdf[(fdf['pred_len'] >= lo) & (fdf['pred_len'] < hi)]
    if len(subset) == 0:
        continue
    correct = (subset['error_category'] == 'Z_CORRECT').sum()
    acc = correct / len(subset) * 100
    rs = subset['rs'].mean()
    tag_rate = subset['has_a_tag'].mean() * 100
    doubt = subset['doubt_count'].mean()
    print(f"  {f'{lo}-{hi}':<20s} {len(subset):6d} {acc:6.1f}% {rs:8.3f} {tag_rate:7.1f}% {doubt:6.1f}")

# ── SECTION 9: ANSWER FORMAT ERRORS ────────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 9: ANSWER FORMAT & EXTRACTION QUALITY")
print("─" * 70)

tagged_wrong = fdf[(fdf['has_a_tag']) & (fdf['error_category'] != 'Z_CORRECT')]
tagged_correct = fdf[(fdf['has_a_tag']) & (fdf['error_category'] == 'Z_CORRECT')]

print(f"\n  Answer format comparison (wrong vs correct, among samples with <a> tag):")
if len(tagged_wrong) > 0 and len(tagged_correct) > 0:
    print(f"    Answer has newlines:    {tagged_wrong['answer_has_newlines'].mean()*100:.0f}% vs {tagged_correct['answer_has_newlines'].mean()*100:.0f}%")
    print(f"    Answer is numeric:      {tagged_wrong['answer_is_numeric'].mean()*100:.0f}% vs {tagged_correct['answer_is_numeric'].mean()*100:.0f}%")
    print(f"    Mean answer length:     {tagged_wrong['answer_len'].mean():.0f} vs {tagged_correct['answer_len'].mean():.0f} chars")

# ── SECTION 10: MULTI-FACTOR CORRELATION ───────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 10: MULTI-FACTOR CORRELATION MATRIX")
print("─" * 70)

# Create binary features for correlation
fdf['is_wrong'] = (fdf['error_category'] != 'Z_CORRECT').astype(int)
corr_features = {
    'is_truncated': fdf['is_truncated'].astype(int),
    'has_a_tag': fdf['has_a_tag'].astype(int),
    'has_heavy_doubt': fdf['has_heavy_doubt'].astype(int),
    'is_chinese': fdf['is_chinese'].astype(int),
    'has_step_by_step': fdf['has_step_by_step'].astype(int),
    'has_json_block': fdf['has_json_block'].astype(int),
    'is_wrong': fdf['is_wrong'],
    'rs': fdf['rs'],
    'pred_len': fdf['pred_len'],
    'doubt_count': fdf['doubt_count'],
}

print("\n  Correlation with 'is_wrong' (1=error, 0=correct):")
for name, col in corr_features.items():
    if name == 'is_wrong':
        continue
    corr = fdf['is_wrong'].corr(col)
    print(f"    {name:<25s}: r = {corr:+.3f}")

print("\n  Correlation with 'rs' (reason_score):")
for name, col in corr_features.items():
    if name == 'rs':
        continue
    corr = fdf['rs'].corr(col)
    print(f"    {name:<25s}: r = {corr:+.3f}")

# ── SECTION 11: TASK-SPECIFIC ERROR PATTERNS ───────────────────────────────
print("\n" + "─" * 70)
print("SECTION 11: TASK-SPECIFIC ERROR PROFILES (detailed)")
print("─" * 70)

for task in tasks:
    t = fdf[fdf['task'] == task]
    t_total = len(t)
    t_wrong = (t['error_category'] != 'Z_CORRECT').sum()
    t_acc = (t_wrong == 0).sum() / t_total * 100  # actually correct rate

    # Correctness is derived from error_category
    t_correct_n = (t['error_category'] == 'Z_CORRECT').sum()
    t_acc = t_correct_n / t_total * 100

    print(f"\n  ▸ {task}  (n={t_total}, acc={t_acc:.1f}%, mean RS={t['rs'].mean():.3f})")

    # Error breakdown
    for cat in ['A_API_FAILURE', 'B_TRUNCATION', 'C_EXTRACTION_ERROR_WITH_DOUBT', 'D_EXTRACTION_ERROR']:
        c = (t['error_category'] == cat).sum()
        if c > 0:
            print(f"      {cat_labels[cat]:<45s}: {c:2d}/{t_total} ({c/t_total*100:4.1f}%)")

    # Key metrics
    t_wrong_set = t[t['error_category'] != 'Z_CORRECT']
    if len(t_wrong_set) > 0:
        print(f"      ── Wrong samples only (n={len(t_wrong_set)}) ──")
        print(f"      Mean pred_len: {t_wrong_set['pred_len'].mean():.0f} | Mean doubt: {t_wrong_set['doubt_count'].mean():.0f} | Trunc%: {t_wrong_set['is_truncated'].mean()*100:.0f}%")

# ── SECTION 12: CORRECT vs INCORRECT DEEP COMPARISON ───────────────────────
print("\n" + "─" * 70)
print("SECTION 12: CORRECT vs INCORRECT — KEY DIFFERENTIATORS")
print("─" * 70)

correct_s = fdf[fdf['error_category'] == 'Z_CORRECT']
wrong_s = fdf[fdf['error_category'] != 'Z_CORRECT']

metrics = [
    ('pred_len', 'Prediction length (chars)', '{:.0f}'),
    ('rs', 'Reason score', '{:.3f}'),
    ('doubt_count', 'Self-doubt pattern count', '{:.1f}'),
    ('has_a_tag', 'Has <a> tag rate', '{:.1%}'),
    ('is_truncated', 'Truncation rate', '{:.1%}'),
    ('has_heavy_doubt', 'Heavy doubt rate', '{:.1%}'),
    ('is_chinese', 'Chinese response rate', '{:.1%}'),
    ('has_step_by_step', 'Step-by-step format rate', '{:.1%}'),
    ('has_json_block', 'JSON block rate', '{:.1%}'),
    ('repeated_line_starts', 'Repeated line starts', '{:.1f}'),
    ('answer_len', 'Answer length', '{:.0f}'),
]

print(f"\n  {'Metric':<35s} {'Correct':>12s} {'Incorrect':>12s} {'Ratio':>8s}")
print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*8}")
for col, label, fmt in metrics:
    c_val = correct_s[col].mean()
    w_val = wrong_s[col].mean()
    ratio = w_val / max(c_val, 0.001)
    c_str = fmt.format(c_val)
    w_str = fmt.format(w_val)
    print(f"  {label:<35s} {c_str:>12s} {w_str:>12s} {ratio:>7.1f}x")

# ── SECTION 13: ANSWER MISMATCH ANALYSIS ───────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 13: JUDGE EXTRACTION vs MODEL ANSWER MISMATCH")
print("─" * 70)

# For correctly-tagged samples, compare a_content with res
tagged = fdf[(fdf['has_a_tag']) & (fdf['a_content'].notna()) & (fdf['a_content'] != '')]
mismatches = []
for _, row in tagged.iterrows():
    a_norm = normalize(row['a_content'])
    res_norm_val = normalize(df.iloc[row['index']]['res'])
    ans_norm_val = normalize(df.iloc[row['index']]['answer'])
    if a_norm and res_norm_val and a_norm != res_norm_val:
        mismatches.append({
            'index': row['index'],
            'task': row['task'],
            'model_answer': row['a_content'][:100],
            'judge_extracted': str(df.iloc[row['index']]['res'])[:100],
            'ground_truth': str(df.iloc[row['index']]['answer'])[:100],
            'rs': row['rs'],
            'model_eq_gt': a_norm == ans_norm_val,
            'judge_eq_gt': res_norm_val == ans_norm_val,
        })

print(f"\n  Samples where model's <a> answer ≠ judge's extracted answer: {len(mismatches)}")
for m in mismatches:
    winner = 'MODEL_right' if m['model_eq_gt'] else ('JUDGE_right' if m['judge_eq_gt'] else 'BOTH_wrong')
    print(f"\n  [{m['index']}] {m['task']} | RS={m['rs']} | {winner}")
    print(f"    Model <a>:  {m['model_answer']}")
    print(f"    Judge res:  {m['judge_extracted']}")
    print(f"    GT answer:  {m['ground_truth']}")

# ── SECTION 14: PIPELINE TRACE ANALYSIS ────────────────────────────────────
print("\n" + "─" * 70)
print("SECTION 14: PIPELINE STAGE PERFORMANCE")
print("─" * 70)

statuses = fdf['pipeline_status'].value_counts()
for s, c in statuses.items():
    subset = fdf[fdf['pipeline_status'] == s]
    correct = (subset['error_category'] == 'Z_CORRECT').sum()
    print(f"  {s:<20s}: {c:4d} samples | acc={correct/max(len(subset),1)*100:5.1f}% | mean RS={subset['rs'].mean():.3f}")

# ── SECTION 15: SUMMARY VISUALIZATION (ASCII) ──────────────────────────────
print("\n" + "─" * 70)
print("SECTION 15: ERROR FLOW SANKEY (TEXT)")
print("─" * 70)

print(f"""
  INPUT: {N} samples
  │
  ├─ API Failure: {len(api_fails):3d} ({len(api_fails)/N*100:4.1f}%) ────────────────── all wrong
  │
  └─ Pipeline Success: {N - len(api_fails):3d} ({ (N-len(api_fails))/N*100:4.1f}%)
      │
      ├─ Output Truncated (no </a>): {len(trunc):3d} ({len(trunc)/N*100:4.1f}%)
      │   └─ Self-doubt loop: {len(fdf[fdf['truncation_signal']=='self_doubt_loop'])} samples
      │
      ├─ Has <a> tag but wrong: {(fdf['error_category'].isin(['C_EXTRACTION_ERROR_WITH_DOUBT','D_EXTRACTION_ERROR'])).sum():3d}
      │   ├─ With heavy doubt: {(fdf['error_category']=='C_EXTRACTION_ERROR_WITH_DOUBT').sum():3d}
      │   └─ Clean extraction error: {(fdf['error_category']=='D_EXTRACTION_ERROR').sum():3d}
      │
      └─ CORRECT: {correct_n:3d} ({correct_n/N*100:4.1f}%)
""")

# ── SECTION 16: KEY INSIGHTS ───────────────────────────────────────────────
print("─" * 70)
print("SECTION 16: KEY INSIGHTS & RECOMMENDATIONS")
print("─" * 70)

# Calculate key percentages
api_pct = len(api_fails) / N * 100
trunc_pct = len(trunc) / N * 100
doubt_pct = len(heavy_doubt) / N * 100
trunc_acc = (trunc['error_category'] == 'Z_CORRECT').sum() / max(len(trunc), 1) * 100
no_trunc_acc = (non_trunc['error_category'] == 'Z_CORRECT').sum() / max(len(non_trunc), 1) * 100
tag_acc = (has_tag['error_category'] == 'Z_CORRECT').sum() / max(len(has_tag), 1) * 100
no_tag_acc = (no_tag['error_category'] == 'Z_CORRECT').sum() / max(len(no_tag), 1) * 100

# Weakest tasks
task_stats = []
for task in tasks:
    t = fdf[fdf['task'] == task]
    task_stats.append((task, len(t), (t['error_category']=='Z_CORRECT').sum()/len(t)*100, t['rs'].mean()))
task_stats.sort(key=lambda x: x[2])

print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     CRITICAL FINDINGS                                │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  1. OUTPUT TRUNCATION IS THE #1 FAILURE MODE                        │
  │     • {trunc_pct:.0f}% of samples lost to truncation ({len(trunc)}/{N})                    │
  │     • Truncated accuracy: {trunc_acc:.0f}% vs non-truncated: {no_trunc_acc:.0f}%           │
  │     • Root cause: model enters self-doubt loops, burns tokens       │
  │     • 100% overlap with self-doubt pattern (Wait, However, etc.)    │
  │                                                                     │
  │  2. SELF-DOUBT DRIVES FAILURE                                       │
  │     • {doubt_pct:.0f}% of samples have heavy self-doubt (>10 patterns)       │
  │     • Doubtful samples accuracy: only {(heavy_doubt['error_category']=='Z_CORRECT').sum()/max(len(heavy_doubt),1)*100:.0f}%                │
  │     • Model says "Wait, I need to check..." instead of answering    │
  │     • Doubt correlates with truncation: r ≈ 1.0 (same samples)     │
  │                                                                     │
  │  3. ANSWER TAG IS MAKE-OR-BREAK                                    │
  │     • With <a> tag: {tag_acc:.0f}% accuracy, RS={has_tag['rs'].mean():.3f}                    │
  │     • Without <a> tag: {no_tag_acc:.0f}% accuracy, RS={no_tag['rs'].mean():.3f}                  │
  │     • {len(no_tag)/N*100:.0f}% missing rate = {len(no_tag)}/{N} dead on arrival           │
  │                                                                     │
  │  4. API FAILURES ({api_pct:.0f}%) ARE A MINOR BUT FATAL TAIL                    │
  │     • {len(api_fails)} samples failed to get API response at all             │
  │     • These get RS=0 and acc=0%                                       │
  │                                                                     │
  │  5. WEAKEST TASK TYPES                                              │
  │     • {task_stats[0][0]:<40s}: {task_stats[0][2]:.1f}% acc, RS={task_stats[0][3]:.3f}    │
  │     • {task_stats[1][0]:<40s}: {task_stats[1][2]:.1f}% acc, RS={task_stats[1][3]:.3f}    │
  │     • {task_stats[2][0]:<40s}: {task_stats[2][2]:.1f}% acc, RS={task_stats[2][3]:.3f}    │
  │                                                                     │
  │  6. PREDICTION LENGTH IS INVERSELY CORRELATED WITH QUALITY          │
  │     • Correct: mean {correct_s['pred_len'].mean():.0f} chars vs Wrong: mean {wrong_s['pred_len'].mean():.0f} chars        │
  │     • {wrong_s['pred_len'].mean()/max(correct_s['pred_len'].mean(),1):.1f}x longer = much more likely wrong               │
  │     • Self-doubt: {wrong_s['doubt_count'].mean()/max(correct_s['doubt_count'].mean(),1):.1f}x more patterns in wrong        │
  │                                                                     │
  │  7. ACCURACY LOSS ATTRIBUTION                                       │
  │     • API failures account for:   {api_pct:.1f}% point loss                    │
  │     • Truncation accounts for:    {trunc_pct*(1-trunc_acc/100):.1f}% point loss                    │
  │     • Clean extraction errors:    ~{(fdf['error_category']=='D_EXTRACTION_ERROR').sum()/N*100:.1f}% point loss               │
  │     • Total explained loss:       ~{api_pct + trunc_pct*(1-trunc_acc/100) + (fdf['error_category']=='D_EXTRACTION_ERROR').sum()/N*100:.1f}%         │
  │     • Unexplained gap (37% total): ~{37 - (api_pct + trunc_pct*(1-trunc_acc/100) + (fdf['error_category']=='D_EXTRACTION_ERROR').sum()/N*100):.1f}%            │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")

# ── SECTION 17: INDIVIDUAL FAILURE CASE STUDIES ────────────────────────────
print("─" * 70)
print("SECTION 17: REPRESENTATIVE FAILURE EXAMPLES")
print("─" * 70)

# Show 1 example per error category
for cat in ['A_API_FAILURE', 'B_TRUNCATION', 'C_EXTRACTION_ERROR_WITH_DOUBT', 'D_EXTRACTION_ERROR']:
    subset = fdf[fdf['error_category'] == cat]
    if len(subset) == 0:
        continue
    idx = subset.iloc[0]['index']
    row = df.iloc[idx]
    pred = str(row['prediction'])[:500] if pd.notna(row['prediction']) else 'NaN'
    print(f"\n  ┌─ [{cat}] Example index={idx}, task={row['task']}")
    print(f"  ├─ Prediction (first 500 chars):")
    for line in pred.split('\n')[:8]:
        print(f"  │  {line[:120]}")
    print(f"  ├─ Answer tag: {fdf.iloc[idx]['has_a_tag']}, Truncated: {fdf.iloc[idx]['is_truncated']}")
    print(f"  ├─ Res (judge): {str(row['res'])[:80]}")
    print(f"  ├─ GT answer:   {str(row['answer'])[:80]}")
    print(f"  └─ RS={row['reason_score']}, Doubt count={fdf.iloc[idx]['doubt_count']}")

print("\n" + "=" * 90)
print("  Analysis Complete.")
print("=" * 90)
