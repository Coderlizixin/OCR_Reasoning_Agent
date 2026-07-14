import pandas as pd
import numpy as np

df = pd.read_excel(r'T:\PyCharmProject\OCR-Reasoning\OCR-Reasoning-main\outputs\Qwen3.5-9B-OCRAgentV3\T20260711_G86c9902b\Qwen3.5-9B-OCRAgentV3_OCR_Reasoning_qwen-plus.xlsx')

# Check the task column
print("=== Task column values ===")
print(df['task'].value_counts())
print()

# Check the question_type column
print("=== question_type column sample ===")
print(df['question_type'].head(20))
print()

# Use task for breakdown
print("=== Breakdown by task ===")
for task in sorted(df['task'].unique()):
    sub = df[df['task'] == task]
    n_correct = 0
    for i in sub.index:
        res = str(sub.loc[i, 'res']).strip().lower()
        ans = str(sub.loc[i, 'answer']).strip().lower()
        if res == ans:
            n_correct += 1
    avg_score = sub['reason_score'].mean()
    n_total = len(sub)
    print(f"{task}: correct={n_correct}/{n_total} ({100*n_correct/n_total:.1f}%), avg_reason={avg_score:.3f}")

# Analyze reason_score distribution for correct vs wrong
print("\n=== Reason Score: Correct vs Wrong ===")
correct_mask = []
for i in range(len(df)):
    res = str(df.iloc[i]['res']).strip().lower()
    ans = str(df.iloc[i]['answer']).strip().lower()
    correct_mask.append(res == ans)

correct_scores = df[np.array(correct_mask)]['reason_score']
wrong_scores = df[~np.array(correct_mask)]['reason_score']
print(f"Correct answers: mean={correct_scores.mean():.3f}, median={correct_scores.median():.3f}, min={correct_scores.min():.3f}, max={correct_scores.max():.3f}")
print(f"Wrong answers: mean={wrong_scores.mean():.3f}, median={wrong_scores.median():.3f}, min={wrong_scores.min():.3f}, max={wrong_scores.max():.3f}")

# Language breakdown
print("\n=== Language breakdown ===")
for lang in sorted(df['language'].unique()):
    sub = df[df['language'] == lang]
    n_correct = 0
    for i in sub.index:
        res = str(sub.loc[i, 'res']).strip().lower()
        ans = str(sub.loc[i, 'answer']).strip().lower()
        if res == ans:
            n_correct += 1
    print(f"{lang}: correct={n_correct}/{len(sub)} ({100*n_correct/len(sub):.1f}%), avg_reason={sub['reason_score'].mean():.3f}")

# Analyze the log column for more detail
print("\n=== Log column analysis ===")
# Check for V3-specific logs in the 'log' column
for i in range(min(10, len(df))):
    log_content = str(df.iloc[i]['log'])
    # Find V3-related content
    has_v3 = 'V3' in log_content or 'Phase' in log_content
    print(f"[{i}] has_V3_logs={has_v3}, log_len={len(log_content)}")
    if has_v3 and len(log_content) > 50:
        # Print first 200 chars
        print(f"    {log_content[:200]}")
