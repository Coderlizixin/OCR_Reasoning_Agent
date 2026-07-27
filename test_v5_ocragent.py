"""
Quick test for OCRAgentV5: Planning → Executor (Recursive Decomposition) → Reasoning.

Tests V5 on a single image + question. Uses the same API endpoint as V4.
Run: python test_v5_ocragent.py
"""
import json
import os
import sys
import time as _time

from vlmeval.vlm.ocr_agent_v5 import OCRAgentV5

# ==========================================
# 0. Configuration
# ==========================================
IMAGE_PATH = "test_case/test_images/image26.jpg"
QUESTION = "What is the value of the largest bar in the chart?"
# Alternative test cases (uncomment to test):
# IMAGE_PATH = "test_case/test_images/image14.jpg"
# QUESTION = "What is the total price of the foods whose names end with the letter 'r'?"

API_BASE = 'http://10.3.112.34:22630/v1/chat/completions'
MODEL_NAME = 'Qwen3.5-9B'

OUTPUT_DIR = "test_case/v5_test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("OCRAgentV5 — Quick Test")
    print("=" * 70)
    print(f"Image:  {IMAGE_PATH}")
    print(f"Question: {QUESTION}")
    print(f"API:    {API_BASE}")
    print(f"Model:  {MODEL_NAME}")
    print("=" * 70)

    # Build the model
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
        retry=10,
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

    # Build input (matching the format from run.py / inference.py)
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

    # Show trace summary
    trace = agent.last_trace
    if trace:
        print(f"Status:       {trace.get('overall_status', '?')}")
        print(f"Total time:   {trace.get('total_time_s', '?')}s")

        planning = trace.get('planning', {})
        print(f"Planning:     {planning.get('status', '?')} "
              f"({planning.get('n_steps', 0)} steps, {planning.get('time_s', 0)}s)")
        if planning.get('strategy'):
            print(f"Strategy:     {planning['strategy'][:150]}")

        executor = trace.get('executor', {})
        print(f"Executor:     {executor.get('overall_status', '?')} "
              f"({executor.get('n_ok', 0)}/{executor.get('n_total', 0)} OK, "
              f"{executor.get('time_s', 0)}s)")

        for sid, sdata in executor.get('steps', {}).items():
            status = sdata.get('status', '?')
            stype = sdata.get('type', '?')
            retries = sdata.get('retry_count', 0)
            extra = ""
            if sdata.get('decomposed'):
                extra = f" [decomposed: {sdata.get('n_sub_steps', '?')} sub-steps, "
                extra += f"{sdata.get('n_sub_ok', '?')} OK]"
            if status == 'ok':
                result = sdata.get('result', '')[:120]
                print(f"  [{sid}] {stype}: OK (retries={retries}){extra} → {result}")
            else:
                error = sdata.get('error', '?')[:120]
                print(f"  [{sid}] {stype}: FAILED (retries={retries}){extra} → {error}")

        reasoning = trace.get('reasoning', {})
        print(f"Reasoning:    has_answer={reasoning.get('has_answer', False)} "
              f"({reasoning.get('time_s', 0)}s)")

    # Write full trace to file
    trace_path = os.path.join(OUTPUT_DIR, 'last_trace.json')
    with open(trace_path, 'w', encoding='utf-8') as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    print(f"\nFull trace saved to: {trace_path}")

    # Also check the trace log (jsonl)
    log_path = os.path.join(OUTPUT_DIR, 'execution_traces.jsonl')
    if os.path.exists(log_path):
        print(f"Trace log:      {log_path}")


if __name__ == '__main__':
    main()
