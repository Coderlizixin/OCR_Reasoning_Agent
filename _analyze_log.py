import re
from collections import Counter, defaultdict

log_path = r'T:\PyCharmProject\OCR-Reasoning\OCR-Reasoning-main\test_case\log\pipeline.log'
with open(log_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Extract V3-specific patterns
v3_phase1_start = 0
v3_phase1_ok = 0
v3_phase1_fail = 0
v3_phase1_json_fail = 0
v3_phase2_done = 0
v3_phase2_stats = []
v3_grounding_valid = []
v3_grounding_total = 0
v3_grounding_done = 0
v3_understand_done = 0
v3_understand_fail = 0
v3_compute_llm = 0
v3_logical_llm = 0
v3_phase3_start = 0
v3_phase3_fallback = 0

for line in lines:
    line = line.strip()
    if '[V3 Phase1] Structured Planning start' in line:
        v3_phase1_start += 1
    if '[V3 PLANNING_OK]' in line:
        v3_phase1_ok += 1
    if '[V3 PLANNING_FAIL]' in line:
        v3_phase1_fail += 1
    if '[V3 Phase1] JSON parse failed' in line:
        v3_phase1_json_fail += 1
    if '[V3 Phase2] Done' in line:
        v3_phase2_done += 1
        # Parse: "Done in 69s: 0 done, 1 failed, 3 pending | 1 grounding, 0 understanding, ..."
        m = re.search(r'(\d+) done, (\d+) failed, (\d+) pending \| (\d+) grounding, (\d+) understanding', line)
        if m:
            v3_phase2_stats.append({
                'done': int(m.group(1)), 'failed': int(m.group(2)),
                'pending': int(m.group(3)), 'grounding': int(m.group(4)),
                'understanding': int(m.group(5))
            })
    if '[V3 Grounding]' in line and 'valid bboxes' in line:
        m = re.search(r'(\d+) valid bboxes', line)
        if m:
            v3_grounding_valid.append(int(m.group(1)))
    if '[V3 Grounding]' in line and ('done' in line or 's' in line):
        v3_grounding_done += 1
    if '[V3 Understand]' in line and 'done' in line:
        v3_understand_done += 1
    if '[V3 Compute]' in line and 'safe_eval failed' in line:
        v3_compute_llm += 1
    if '[V3 Logical]' in line and 'safe_eval failed' in line:
        v3_logical_llm += 1
    if '[V3 Phase3] Final Reasoning start' in line:
        v3_phase3_start += 1
    if '[V3 Phase3] Fallback Reasoning start' in line:
        v3_phase3_fallback += 1

print(f"\n=== V3 Planning Stats ===")
print(f"Phase1 starts: {v3_phase1_start}")
print(f"PLANNING_OK: {v3_phase1_ok}")
print(f"PLANNING_FAIL: {v3_phase1_fail}")
print(f"JSON parse failed: {v3_phase1_json_fail}")
if v3_phase1_start > 0:
    print(f"Success rate: {v3_phase1_ok}/{v3_phase1_start} = {100*v3_phase1_ok/v3_phase1_start:.1f}%")

print(f"\n=== V3 Phase 2 Stats ===")
print(f"Phase2 completions: {v3_phase2_done}")
if v3_phase2_stats:
    total_done = sum(s['done'] for s in v3_phase2_stats)
    total_failed = sum(s['failed'] for s in v3_phase2_stats)
    total_pending = sum(s['pending'] for s in v3_phase2_stats)
    total_grounding = sum(s['grounding'] for s in v3_phase2_stats)
    total_understanding = sum(s['understanding'] for s in v3_phase2_stats)
    print(f"Total steps done: {total_done}")
    print(f"Total steps failed: {total_failed}")
    print(f"Total steps pending: {total_pending}")
    print(f"Total grounding calls: {total_grounding}")
    print(f"Total understanding calls: {total_understanding}")

    # Average per question
    n = len(v3_phase2_stats)
    print(f"\nPer question (avg):")
    print(f"  done: {total_done/n:.1f}")
    print(f"  failed: {total_failed/n:.1f}")
    print(f"  pending: {total_pending/n:.1f}")
    print(f"  grounding: {total_grounding/n:.1f}")
    print(f"  understanding: {total_understanding/n:.1f}")

    # How many questions have steps that failed
    questions_with_failures = sum(1 for s in v3_phase2_stats if s['failed'] > 0)
    print(f"\nQuestions with >=1 step failure: {questions_with_failures}/{n} ({100*questions_with_failures/n:.1f}%)")

    # How many have ZERO done
    zero_done = sum(1 for s in v3_phase2_stats if s['done'] == 0)
    print(f"Questions with 0 done steps: {zero_done}/{n} ({100*zero_done/n:.1f}%)")

print(f"\n=== V3 Grounding Stats ===")
print(f"Grounding 'done' lines: {v3_grounding_done}")
if v3_grounding_valid:
    print(f"Valid bbox counts: min={min(v3_grounding_valid)}, max={max(v3_grounding_valid)}, avg={sum(v3_grounding_valid)/len(v3_grounding_valid):.1f}")
    # Distribution
    bins = [(0,5), (5,10), (10,20), (20,50), (50,100), (100,999)]
    for lo, hi in bins:
        cnt = sum(1 for x in v3_grounding_valid if lo <= x < hi)
        print(f"  [{lo}-{hi}): {cnt}")

print(f"\n=== V3 Understanding Stats ===")
print(f"Understanding done: {v3_understand_done}")

print(f"\n=== V3 Compute/Logical Fallback Stats ===")
print(f"Compute safe_eval failed -> LLM: {v3_compute_llm}")
print(f"Logical safe_eval failed -> LLM: {v3_logical_llm}")
print(f"Total safe_eval failures: {v3_compute_llm + v3_logical_llm}")

print(f"\n=== V3 Phase 3 Stats ===")
print(f"Phase3 starts: {v3_phase3_start}")
print(f"Phase3 fallback starts: {v3_phase3_fallback}")

# Check bbox truncation
bbox_trunc = sum(1 for line in lines if '超过上限' in line and '截断' in line)
print(f"\n=== Bbox Truncation ===")
print(f"Total bbox truncation events: {bbox_trunc}")

# Check for empty/invalid grounding results
print(f"\n=== Grounding Quality ===")
valid_bboxes_0 = sum(1 for x in v3_grounding_valid if x == 0)
valid_bboxes_too_many = sum(1 for x in v3_grounding_valid if x > 8)
print(f"Grounding with 0 valid bboxes: {valid_bboxes_0}")
print(f"Grounding with >8 valid bboxes: {valid_bboxes_too_many}")

# Count unique questions
v3_q_starts = [line for line in lines if '[V3 Phase1] Structured Planning start' in line]
print(f"\n=== Total V3 Questions Processed ===")
print(f"Phase1 start count: {len(v3_q_starts)}")
