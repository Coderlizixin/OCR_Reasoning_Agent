import pandas as pd
import numpy as np

df = pd.read_excel(r'T:\PyCharmProject\OCR-Reasoning\OCR-Reasoning-main\outputs\Qwen3.5-9B-OCRAgentV3\T20260711_G86c9902b\Qwen3.5-9B-OCRAgentV3_OCR_Reasoning_qwen-plus.xlsx')

print("=== res column value counts (top 20) ===")
print(df['res'].value_counts().head(20))
print()

# Check empty predictions
print("=== Empty/null predictions ===")
empty_pred = df['prediction'].isna() | (df['prediction'] == '') | (df['prediction'] == 'nan')
print(f'Empty predictions: {empty_pred.sum()}')

# Check empty res
print("\n=== Empty/null res ===")
empty_res = df['res'].isna() | (df['res'] == '') | (df['res'] == 'nan')
print(f'Empty res: {empty_res.sum()}')

# Compare res vs answer
print("\n=== res vs answer (first 30) ===")
for i in range(min(30, len(df))):
    pred_short = str(df.iloc[i]['prediction'])[:120].replace('\n', ' ')
    res_val = str(df.iloc[i]['res'])
    ans_val = str(df.iloc[i]['answer'])
    score = df.iloc[i]['reason_score']
    print(f'[{i}] res={res_val} | answer={ans_val} | score={score}')
    print(f'    pred: {pred_short}')
    print()

# Check for cases where res matches answer exactly
print("\n=== Match analysis ===")
matches = []
for i in range(len(df)):
    res_str = str(df.iloc[i]['res']).strip().lower()
    ans_str = str(df.iloc[i]['answer']).strip().lower()
    matches.append(res_str == ans_str)
print(f'Exact string matches (case-insensitive, stripped): {sum(matches)}/{len(df)}')

# Check log column for errors
print("\n=== log column error patterns ===")
error_count = 0
for i in range(len(df)):
    log_val = str(df.iloc[i]['log'])
    if 'error' in log_val.lower() or 'fail' in log_val.lower() or 'timeout' in log_val.lower():
        error_count += 1
print(f'Rows with error/fail/timeout in log: {error_count}')

# Check question_type breakdown with correct/wrong
print("\n=== Detailed breakdown by question_type ===")
for qt in sorted(df['question_type'].unique()):
    sub = df[df['question_type'] == qt]
    matched = 0
    for i in range(len(sub)):
        res_str = str(sub.iloc[i]['res']).strip().lower()
        ans_str = str(sub.iloc[i]['answer']).strip().lower()
        if res_str == ans_str:
            matched += 1
    avg_score = sub['reason_score'].mean()
    print(f'{qt}: match={matched}/{len(sub)} ({100*matched/len(sub):.1f}%), avg_reason={avg_score:.3f}')

# Analyze prediction length
print("\n=== Prediction length stats ===")
pred_lens = df['prediction'].apply(lambda x: len(str(x)))
print(pred_lens.describe())

# Check very short predictions (possible empty outputs)
short_preds = df[pred_lens < 20]
print(f"\nPredictions < 20 chars: {len(short_preds)}")
for i in short_preds.index[:10]:
    print(f'  [{i}] pred={df.iloc[i]["prediction"]} | res={df.iloc[i]["res"]} | answer={df.iloc[i]["answer"]}')

# Check log for fallback logs
print("\n=== Fallback detection in log ===")
fallback_count = 0
for i in range(len(df)):
    log_val = str(df.iloc[i]['log'])
    if 'PLANNING_FAIL' in log_val or 'FALLBACK' in log_val.upper():
        fallback_count += 1
print(f'Rows with PLANNING_FAIL or FALLBACK in log: {fallback_count}')
