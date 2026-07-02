"""Quick re-test of sample 13 with improved prompts."""
import os, json, pandas as pd, time
from vlmeval.config import supported_VLM
from vlmeval.vlm.reasoning_trace_ocragent import ReasoningTraceOCRAgent

TEST_IDX = 13
VARIANTS = [
    'Qwen3.5-9B-ReasoningTrace',
    'Qwen3.5-9B-ReasoningTrace-OV',
    'Qwen3.5-9B-ReasoningTrace-OR',
]

tsv_path = r'C:\Users\19639\LMUData\OCR_Reasoning.tsv'
df = pd.read_csv(tsv_path, sep='\t')

for vname in VARIANTS:
    print('='*60)
    print(f'  {vname}')
    print('='*60)
    model = supported_VLM[vname]()
    row = df[df['index'] == TEST_IDX].iloc[0]
    question = str(row['question'])
    image_path = f'C:\\Users\\19639\\LMUData\\images\\OCR_Reasoning\\{TEST_IDX}.jpg'
    inputs = [
        {'type': 'image', 'value': image_path},
        {'type': 'text', 'value': question},
    ]
    print(f'Q: {question}')
    try:
        ret, answer, resp = model.generate_inner(inputs)
        ans_len = len(str(answer))
        clean = ReasoningTraceOCRAgent._extract_final_answer(answer)
        has_final = 'FINAL_ANSWER' in str(answer)
        print(f'ret={ret}, len={ans_len}, has_FINAL_ANSWER={has_final}')
        print(f'extracted: {repr(clean)}')
        if ans_len > 300:
            print(f'Last 300 chars: ...{str(answer)[-300:]}')
        else:
            print(f'Full: {answer}')
    except Exception as e:
        print(f'EXCEPTION: {e}')
        import traceback
        traceback.print_exc()
    del model
    time.sleep(2)
