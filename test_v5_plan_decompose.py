"""
Quick test for OCRAgentV5 Planning Decomposition.
Uses a known planning-failure question to trigger the decomposer.

Run: python test_v5_plan_decompose.py
"""
import json
import os
import sys
import time as _time

from vlmeval.vlm.ocr_agent_v5 import OCRAgentV5

# ==========================================
# 0. Configuration — using known planning-failure question
# ==========================================
IMAGE_PATH = r"C:\Users\19639\LMUData\images\OCR_Reasoning\17.jpg"
QUESTION = "该图里中国到美国的电子密度的平均值是多少?\n通过一步一步的推理来解决这个复杂的问题。你的回答应该遵守: '数字(保留两位小数)'."

# Fallback test image if LMUData image not available
if not os.path.exists(IMAGE_PATH):
    IMAGE_PATH = "test_case/test_images/image26.jpg"
    QUESTION = "What is the value of the largest bar in the chart?"

API_BASE = 'http://10.3.112.34:22630/v1/chat/completions'
MODEL_NAME = 'Qwen3.5-9B'

OUTPUT_DIR = "test_case/v5_plan_decompose_test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("OCRAgentV5 — Planning Decomposition Test")
    print("=" * 70)
    print(f"Image:  {os.path.basename(IMAGE_PATH)}")
    print(f"Question: {QUESTION[:120]}")
    print(f"API:    {API_BASE}")
    print("=" * 70)

    agent = OCRAgentV5(
        model=MODEL_NAME,
        api_base=API_BASE,
        key='sk-EMPTY',
        temperature=0.0,
        skip_locate=True,
        cross_validate=True,
        temp_planning=0.1,
        temp_locate=0.1,
        temp_understand=0.2,
        temp_compute=0.1,
        temp_logical=0.1,
        temp_reasoning=0.2,
        retry=5,  # fewer top-level retries
        img_size=-1,
        img_detail='high',
        max_tokens=6144,
        trace_dir=OUTPUT_DIR,
        extra_body={
            "mm_processor_kwargs": {
                "images_kwargs": {
                    "size": {
                        "longest_edge": 8192 * 32 * 32,
                        "shortest_edge": 4096 * 32 * 32,
                    }
                }
            }
        },
    )

    inputs = [
        {'type': 'image', 'value': IMAGE_PATH},
        {'type': 'text', 'value': QUESTION},
    ]

    t0 = _time.time()
    ret, answer, resp = agent.generate_inner(inputs)
    elapsed = _time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"RESULTS (elapsed: {elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"Return code: {ret}")
    print(f"Answer: {answer[:500] if answer else '(empty)'}")
    print()

    trace = agent.last_trace
    if trace:
        status = trace.get('overall_status', '?')
        planning = trace.get('planning', {})
        print(f"Status:         {status}")
        print(f"Planning:       {planning.get('status', '?')}")
        if planning.get('status') == 'planning_decomposed':
            print(f"  Original:     {planning.get('original_status', '?')}")
            print(f"  Sub-questions: {planning.get('n_sub_questions', 0)}")
            for sq in planning.get('sub_questions', []):
                print(f"    [{sq['id']}]: {sq['question'][:100]}")
                print(f"         deps: {sq.get('depends_on', [])}")

            executor = trace.get('executor', {})
            sub_answers = executor.get('sub_answers', {})
            print(f"  Sub-answers:")
            for sid, sa in sub_answers.items():
                print(f"    [{sid}]: status={sa['status']}")
                print(f"           answer={str(sa.get('answer', ''))[:150]}")

        reasoning = trace.get('reasoning', {})
        print(f"Reasoning:      has_answer={reasoning.get('has_answer', False)}")

    # Write trace
    trace_path = os.path.join(OUTPUT_DIR, 'last_trace.json')
    with open(trace_path, 'w', encoding='utf-8') as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    print(f"\nFull trace: {trace_path}")


if __name__ == '__main__':
    main()
