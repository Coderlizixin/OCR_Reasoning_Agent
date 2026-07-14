import pandas as pd
import numpy as np
import re

df = pd.read_excel(r'T:\PyCharmProject\OCR-Reasoning\OCR-Reasoning-main\outputs\Qwen3.5-9B-OCRAgentV3\T20260711_G86c9902b\Qwen3.5-9B-OCRAgentV3_OCR_Reasoning_qwen-plus.xlsx')

print("=" * 80)
print("COMPREHENSIVE ERROR ANALYSIS - V3 BATCH RUN")
print("=" * 80)

# 1. Categorize errors
def is_numeric(s):
    try:
        float(str(s).replace(',', '').replace(' ', '').replace('%', '').replace('$', '').replace('¥', '').replace('元', '').replace('€', ''))
        return True
    except:
        return False

def extract_number(s):
    """Extract the first number from a string"""
    s = str(s).replace(',', '').replace(' ', '')
    match = re.search(r'[-]?\d+\.?\d*', s)
    if match:
        return float(match.group())
    return None

def norm(s):
    return str(s).strip().lower().replace(' ', '').replace(',', '')

# Analyze each row
categories = []
for i in range(len(df)):
    pred = str(df.iloc[i]['prediction'])
    res = str(df.iloc[i]['res'])
    ans = str(df.iloc[i]['answer'])
    reason = str(df.iloc[i]['reasoning'])
    score = df.iloc[i]['reason_score']
    qtype = df.iloc[i]['question_type']

    # Check for empty/truncated output
    if len(pred) < 30:
        cat = 'EMPTY/TRUNCATED'
    elif '!!!' in pred and len(set(pred)) < 10:
        cat = 'CORRUPTED_OUTPUT'
    elif norm(res) == norm(ans):
        cat = 'CORRECT'
    elif res == '' or res == 'nan':
        cat = 'MISSING_ANSWER'
    elif extract_number(res) is not None and extract_number(ans) is not None:
        # Both numeric - check if close
        n_pred = extract_number(res)
        n_ans = extract_number(ans)
        if n_ans == 0:
            cat = 'NUMERIC_MISMATCH'
        else:
            ratio = abs(n_pred - n_ans) / abs(n_ans)
            if ratio < 0.01:
                cat = 'ROUNDING_ERROR'
            elif ratio < 0.1:
                cat = 'CLOSE_BUT_WRONG'
            elif ratio < 1.0:
                cat = 'NUMERIC_MISMATCH'
            else:
                cat = 'WILDLY_OFF'
    else:
        # Text answers
        cat = 'TEXT_MISMATCH'

    categories.append(cat)

df['error_cat'] = categories
print("\n=== Error Category Distribution ===")
for cat, cnt in pd.Series(categories).value_counts().items():
    avg_score = df[np.array(categories) == cat]['reason_score'].mean()
    print(f"  {cat}: {cnt} ({100*cnt/len(df):.1f}%), avg_reason={avg_score:.3f}")

# 2. Analyze CORRECT answers with low reason scores
print("\n=== Correct Answers with Reason Score < 0.5 ===")
low_score_correct = df[(np.array(categories) == 'CORRECT') & (df['reason_score'] < 0.5)]
print(f"Count: {len(low_score_correct)}/{sum(np.array(categories) == 'CORRECT')}")
for i in low_score_correct.index[:5]:
    print(f"  [{i}] q={str(df.iloc[i]['question'])[:80]} | ans={df.iloc[i]['answer']} | score={df.iloc[i]['reason_score']}")

# 3. Analyze WILDLY_OFF cases
print("\n=== WILDLY_OFF Cases (first 10) ===")
wild = df[np.array(categories) == 'WILDLY_OFF']
print(f"Count: {len(wild)}")
for i in wild.index[:10]:
    print(f"  [{i}] q={str(df.iloc[i]['question'])[:80]}")
    print(f"       res={df.iloc[i]['res']} | ans={df.iloc[i]['answer']} | score={df.iloc[i]['reason_score']}")

# 4. Analyze CORRUPTED_OUTPUT
print("\n=== CORRUPTED_OUTPUT Cases ===")
corrupt = df[np.array(categories) == 'CORRUPTED_OUTPUT']
print(f"Count: {len(corrupt)}")
for i in corrupt.index:
    print(f"  [{i}] q={str(df.iloc[i]['question'])[:100]}")
    print(f"       pred starts with: {str(df.iloc[i]['prediction'])[:150]}")

# 5. Analyze by question type and error category
print("\n=== Error Category by Question Type ===")
for qt in sorted(df['question_type'].unique()):
    sub = df[df['question_type'] == qt]
    sub_cats = [categories[i] for i in sub.index]
    correct = sub_cats.count('CORRECT')
    print(f"\n{qt} ({len(sub)} questions):")
    for cat, cnt in pd.Series(sub_cats).value_counts().items():
        print(f"  {cat}: {cnt} ({100*cnt/len(sub):.1f}%)")

# 6. Check spatial reasoning specifically (worst performer at 21.1%)
print("\n=== SPATIAL REASONING - ALL CASES ===")
spatial = df[df['question_type'] == 'spatial reasoning']
for i in spatial.index:
    cat = categories[i]
    print(f"  [{i}] q={str(df.iloc[i]['question'])[:100]}")
    print(f"       res={df.iloc[i]['res']} | ans={df.iloc[i]['answer']} | score={df.iloc[i]['reason_score']} | cat={cat}")

# 7. Check for evidence issues via prediction content
print("\n=== Evidence-related Issues in Predictions ===")
evidence_issues = []
for i in range(len(df)):
    pred = str(df.iloc[i]['prediction']).lower()
    if 'evidence' in pred and ('no ' in pred or 'missing' in pred or 'not provided' in pred or 'not found' in pred):
        evidence_issues.append(i)
print(f"Predictions mentioning missing/no evidence: {len(evidence_issues)}")
for i in evidence_issues[:10]:
    print(f"  [{i}] {str(df.iloc[i]['prediction'])[:150]}")
