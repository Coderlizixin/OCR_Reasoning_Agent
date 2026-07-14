import pandas as pd
import re

df = pd.read_excel(r'outputs\Qwen3.5-9B-OCRAgentV3\T20260711_G86c9902b\Qwen3.5-9B-OCRAgentV3_OCR_Reasoning_qwen-plus.xlsx')

def norm(s):
    return str(s).strip().lower().replace(' ','').replace(',','')

# Find wrong cases
cats = []
for i in range(len(df)):
    res = str(df.iloc[i]['res'])
    ans = str(df.iloc[i]['answer'])
    if norm(res) == norm(ans):
        cats.append('CORRECT')
    else:
        cats.append('WRONG')

print('=== TEXT_MISMATCH samples (first 15) ===')
count = 0
for i in range(len(df)):
    if cats[i] == 'WRONG':
        res = str(df.iloc[i]['res'])
        ans = str(df.iloc[i]['answer'])
        pred = str(df.iloc[i]['prediction'])
        n_r = re.search(r'[-]?\d+\.?\d*', res)
        n_a = re.search(r'[-]?\d+\.?\d*', ans)
        if not (n_r and n_a):
            count += 1
            if count <= 15:
                q = str(df.iloc[i]['question'])[:80]
                print(f'[{i}] Q: {q}')
                print(f'    res: [{res[:120]}]')
                print(f'    ans: [{ans[:80]}]')
                print(f'    pred[:200]: {pred[:200]}')
                print()

print()
print('=== Check for thinking traces in predictions ===')
think_count = 0
for i in range(len(df)):
    p = str(df.iloc[i]['prediction'])[:800]
    if '<｜end▁of▁thinking｜>' in p or '<think>' in p:
        think_count += 1
print(f'Predictions containing thinking tags: {think_count}/{len(df)}')

print()
print('=== Check predictions for <a> tag issues ===')
no_a = 0
for i in range(len(df)):
    if '<a>' not in str(df.iloc[i]['prediction']):
        no_a += 1
print(f'Predictions WITHOUT <a> tags: {no_a}/{len(df)}')

print()
print('=== res column quality ===')
empty_res = 0
long_res = 0
for i in range(len(df)):
    r = str(df.iloc[i]['res']).strip()
    if r in ['', 'nan', 'None']:
        empty_res += 1
    if len(r) > 200:
        long_res += 1
print(f'Empty res: {empty_res}')
print(f'Very long res (>200 chars): {long_res}')

# Show some long res examples
print()
print('=== Long res examples ===')
for i in range(len(df)):
    r = str(df.iloc[i]['res'])
    if len(r) > 200:
        q = str(df.iloc[i]['question'])[:80]
        a = str(df.iloc[i]['answer'])[:80]
        print(f'[{i}] Q: {q}')
        print(f'    ans: [{a}]')
        print(f'    res: [{r[:300]}]')
        print()

# Check if res contains thinking
print()
print('=== res with thinking traces ===')
think_in_res = 0
for i in range(len(df)):
    r = str(df.iloc[i]['res'])
    if '<｜end▁of▁thinking｜>' in r or '<think>' in r:
        think_in_res += 1
print(f'res containing thinking: {think_in_res}')

# Compare res lengths old vs new
print()
print('=== res length stats ===')
res_lens = [len(str(df.iloc[i]['res'])) for i in range(len(df))]
print(f'min={min(res_lens)}, max={max(res_lens)}, avg={sum(res_lens)/len(res_lens):.1f}, median={sorted(res_lens)[len(res_lens)//2]}')

# Check for "Step" or evidence in res (indicating extraction failure)
print()
print('=== res containing Step/evidence artifacts ===')
artifact_count = 0
for i in range(len(df)):
    r = str(df.iloc[i]['res']).lower()
    if 'step s' in r or 'evidence' in r:
        artifact_count += 1
print(f'res with step/evidence artifacts: {artifact_count}')
