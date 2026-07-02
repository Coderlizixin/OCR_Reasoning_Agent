"""
Smoke test: Oracle 实验三模式端到端测试（3 个样本）。
验证 prompt 构建、API 调用、答案提取全流程。
"""
import os
import json
import pandas as pd
import time
from vlmeval.config import supported_VLM

# 测试样本
TEST_INDICES = [13, 17, 21]
VARIANTS = [
    'Qwen3.5-9B-ReasoningTrace',        # oracle_mode=None (default, 修复版)
    'Qwen3.5-9B-ReasoningTrace-OV',     # oracle_mode='values'
    'Qwen3.5-9B-ReasoningTrace-OR',     # oracle_mode='reasoning'
]

# 加载数据集
tsv_path = r"C:\Users\19639\LMUData\OCR_Reasoning.tsv"
df = pd.read_csv(tsv_path, sep='\t')

for variant_name in VARIANTS:
    print(f"\n{'='*70}")
    print(f"  Variant: {variant_name}")
    print(f"{'='*70}")

    model = supported_VLM[variant_name]()

    for idx in TEST_INDICES:
        row = df[df['index'] == idx]
        if row.empty:
            print(f"  idx={idx}: NOT FOUND in dataset")
            continue
        row = row.iloc[0]
        question = str(row['question'])

        # 构建 inputs（模拟框架调用方式）
        image_path = f"C:\\Users\\19639\\LMUData\\images\\OCR_Reasoning\\{idx}.jpg"
        if not os.path.exists(image_path):
            print(f"  idx={idx}: Image not found at {image_path}")
            continue

        inputs = [
            {'type': 'image', 'value': image_path},
            {'type': 'text', 'value': question},
        ]

        print(f"\n  --- idx={idx} ---")
        print(f"  Q: {question[:120]}{'...' if len(question) > 120 else ''}")

        try:
            ret, answer, resp = model.generate_inner(inputs)
            if ret != 0:
                print(f"  ERROR: ret={ret}, answer={str(answer)[:100]}")
                continue

            # 诊断信息
            from vlmeval.vlm.reasoning_trace_ocragent import ReasoningTraceOCRAgent
            clean = ReasoningTraceOCRAgent._extract_final_answer(answer)
            ans_len = len(str(answer))
            print(f"  OK: ret={ret}, len={ans_len}, extracted_answer='{clean}'")

            # 打印 prompt 开头和回答结尾
            if ans_len > 200:
                print(f"  Model output[:200]: {str(answer)[:200]}...")
                print(f"  Model output[-200:]: ...{str(answer)[-200:]}")
            else:
                print(f"  Model output: {answer}")

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(1)

    del model
    time.sleep(2)

print("\n\nDone! All smoke tests completed.")
