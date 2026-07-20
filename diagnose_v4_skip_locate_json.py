"""
快速诊断脚本：在 V4-SkipLocate 上运行 20 道随机题目，捕获每阶段的 JSON 解析成功率。
用法:
    python diagnose_v4_skip_locate_json.py
"""
import os, sys, json, time, random
from collections import Counter

# Read random indices
with open('test_case/random_indices.json', 'r', encoding='utf-8') as f:
    random_indices = json.load(f)

# Pick 20 random indices
random.seed(42)
test_indices = sorted(random.sample(random_indices, 20))
print(f"Testing {len(test_indices)} random indices: {test_indices}")

from vlmeval.vlm.ocr_agent_v4 import OCRAgentV4

agent = OCRAgentV4(
    model='Qwen3.5-9B',
    api_base='http://10.3.112.34:22630/v1/chat/completions',
    key='sk-EMPTY',
    temperature=0.0,
    skip_locate=True,
    temp_planning=0.1,
    temp_understand=0.2,
    temp_compute=0.1,
    temp_logical=0.1,
    temp_reasoning=0.2,
    max_tokens=6144,
    img_size=-1,
    img_detail='high',
    verbose=False,
    retry=3,
    wait=3,
    trace_dir='test_case/v4_skip_locate_diag_traces',
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

# Stats
planning_stats = Counter()
step_stats = {'total': 0, 'locate_parse_err': 0, 'understand_parse_err': 0,
              'compute_parse_err': 0, 'logical_parse_err': 0, 'ok': 0, 'other_err': 0}
overall_stats = Counter()
total_time = 0

for i, idx in enumerate(test_indices):
    image_path = f"C:/Users/19639/LMUData/images/OCR_Reasoning/{idx}.jpg"
    if not os.path.exists(image_path):
        print(f"[{i+1}/{len(test_indices)}] idx={idx}: SKIP (image not found)")
        continue

    # Load question from Excel
    import pandas as pd
    excel_paths = [
        'outputs/Qwen3.5-9B-OCRAgentV4-SkipLocate/Qwen3.5-9B-OCRAgentV4-SkipLocate_OCR_Reasoning_qwen-plus.xlsx',
    ]
    question = f"Question for index {idx}"
    for ep in excel_paths:
        if os.path.exists(ep):
            df = pd.read_excel(ep).set_index('index')
            if idx in df.index:
                question = df.loc[idx, 'question']
                break

    inputs = [
        {'type': 'text', 'value': str(question)},
        {'type': 'image', 'value': image_path},
    ]

    t0 = time.time()
    ret, answer, resp = agent.generate_inner(inputs)
    elapsed = time.time() - t0
    total_time += elapsed

    trace = agent.last_trace
    if trace is None:
        print(f"[{i+1}/{len(test_indices)}] idx={idx}: NO TRACE")
        continue

    status = trace['overall_status']
    overall_stats[status] += 1

    # Planning stage
    plan_status = trace['planning']['status']
    planning_stats[plan_status] += 1
    n_plan_steps = trace['planning']['n_steps']

    # Executor stage
    n_ok = 0
    n_parse_err = 0
    parse_phases = Counter()
    for sid, sd in trace['executor']['steps'].items():
        step_stats['total'] += 1
        if sd.get('status') == 'ok':
            step_stats['ok'] += 1
            n_ok += 1
        else:
            error_msg = sd.get('error', '')
            failed_phase = sd.get('failed_phase', '')
            is_parse_err = any(kw in error_msg.lower() for kw in
                              ['parse', 'json', 'failed to parse', 'jsondecode'])

            if is_parse_err:
                n_parse_err += 1
                if failed_phase == 'locate':
                    step_stats['locate_parse_err'] += 1
                elif failed_phase == 'understand':
                    step_stats['understand_parse_err'] += 1
                elif sd.get('type') == 'compute':
                    step_stats['compute_parse_err'] += 1
                elif sd.get('type') == 'logical':
                    step_stats['logical_parse_err'] += 1
                parse_phases[failed_phase or sd.get('type', '?')] += 1
            else:
                step_stats['other_err'] += 1

    # Has answer?
    has_answer = '<a>' in (answer or '') and '</a>' in (answer or '')

    print(f"[{i+1}/{len(test_indices)}] idx={idx}: status={status} "
          f"plan={plan_status} steps={n_plan_steps} ok={n_ok} err_parse={n_parse_err} "
          f"has_ans={has_answer} time={elapsed:.0f}s")
    if parse_phases:
        print(f"         Parse error phases: {dict(parse_phases)}")

print()
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\nTested: {len(test_indices)} questions, total time: {total_time:.0f}s ({total_time/len(test_indices):.0f}s avg)")
print(f"\nOverall status:")
for s, c in overall_stats.most_common():
    print(f"  {s}: {c} ({100*c/len(test_indices):.1f}%)")

print(f"\nPlanning stage:")
for s, c in planning_stats.most_common():
    print(f"  {s}: {c}")

n_steps_total = step_stats['total']
n_ok = step_stats['ok']
n_parse_err = (step_stats['locate_parse_err'] + step_stats['understand_parse_err'] +
               step_stats['compute_parse_err'] + step_stats['logical_parse_err'])
n_other_err = step_stats['other_err']

print(f"\nExecutor steps (total={n_steps_total}):")
print(f"  OK: {n_ok} ({100*n_ok/n_steps_total:.1f}%)" if n_steps_total else "  OK: 0")
print(f"  JSON parse errors: {n_parse_err} ({100*n_parse_err/n_steps_total:.1f}%)" if n_steps_total else "  JSON parse errors: 0")
print(f"    - Locate phase: {step_stats['locate_parse_err']}")
print(f"    - Understand phase: {step_stats['understand_parse_err']}")
print(f"    - Compute: {step_stats['compute_parse_err']}")
print(f"    - Logical: {step_stats['logical_parse_err']}")
print(f"  Other errors: {n_other_err}")

# Per-question JSON parse failure rate
questions_with_parse_err = 0
for idx in test_indices:
    trace = agent.last_trace
    # We need to re-read from the trace file
print(f"\n=== KEY METRIC ===")
print(f"Steps with JSON parse error: {n_parse_err}/{n_steps_total} = {100*n_parse_err/max(1,n_steps_total):.1f}%")
print(f"(This is the rate at which model thinking-too-long causes incomplete/unparseable JSON output)")
