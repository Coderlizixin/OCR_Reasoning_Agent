import pandas as pd
import re

df = pd.read_excel(r'outputs\Qwen3.5-9B-OCRAgentV3\T20260711_G86c9902b\Qwen3.5-9B-OCRAgentV3_OCR_Reasoning_qwen-plus.xlsx')

def extract_number(s):
    s = str(s).replace(',','').replace(' ','')
    m = re.search(r'[-]?\d+\.?\d*', s)
    return float(m.group()) if m else None

# Categorize wrong answers
wrong_cats = []
for i in range(len(df)):
    res = str(df.iloc[i]['res']).strip()
    ans = str(df.iloc[i]['answer']).strip()

    # Check if actually correct
    res_norm = res.lower().replace(' ', '')
    ans_norm = ans.lower().replace(' ', '')
    if res_norm == ans_norm:
        wrong_cats.append('CORRECT_NORM')
        continue

    n_res = extract_number(res)
    n_ans = extract_number(ans)
    if n_res is not None and n_ans is not None:
        if n_ans != 0 and abs(n_res - n_ans) / abs(n_ans) < 0.01:
            wrong_cats.append('CORRECT_NUMERIC')
            continue

    # Real wrong — categorize
    pred = str(df.iloc[i]['prediction'])

    # Check for thinking loop / excessive repetition
    if len(pred) > 3000 and ('!!!' in pred or pred.count('Let') > 20):
        wrong_cats.append('THINKING_LOOP')
    elif len(pred) < 100:
        wrong_cats.append('TRUNCATED')
    elif n_res is not None and n_ans is not None:
        ratio = abs(n_res - n_ans) / max(abs(n_ans), 1)
        if ratio > 10:
            wrong_cats.append('ORDER_OF_MAGNITUDE_ERROR')
        elif ratio > 2:
            wrong_cats.append('LARGE_NUMERIC_ERROR')
        elif ratio > 0.01:
            wrong_cats.append('MODERATE_NUMERIC_ERROR')
        else:
            wrong_cats.append('OTHER_NUMERIC')
    else:
        # Text mismatch
        if 'judge' in res.lower() or 'model response' in res.lower():
            wrong_cats.append('JUDGE_FAILED_TO_EXTRACT')
        elif len(res) <= 2 and len(ans) >= 2 and ans.isalpha():
            wrong_cats.append('MULTI_SELECT_AS_SINGLE')
        elif res in ['a','b','c','d'] and ans in ['a','b','c','d']:
            wrong_cats.append('WRONG_OPTION')
        else:
            wrong_cats.append('TEXT_MISMATCH')

print('=== Wrong Answer Categories ===')
from collections import Counter
for cat, cnt in Counter(wrong_cats).most_common():
    print(f'  {cat}: {cnt} ({100*cnt/len(df):.1f}%)')

print()

# Show examples of each category
for cat in set(wrong_cats):
    examples = [i for i, c in enumerate(wrong_cats) if c == cat]
    print(f'=== {cat} (first 3) ===')
    for idx in examples[:3]:
        q = str(df.iloc[idx]['question'])[:80]
        res = str(df.iloc[idx]['res'])[:80]
        ans = str(df.iloc[idx]['answer'])[:80]
        pred = str(df.iloc[idx]['prediction'])[:250]
        print(f'  [{idx}] Q: {q}')
        print(f'       res=[{res}]  ans=[{ans}]')
        print(f'       pred: {pred}')
        print()
