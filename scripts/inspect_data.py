#!/usr/bin/env python3
"""Inspect raw data to understand log, prediction, and res columns."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd

XLSX = r"T:\PyCharmProject\OCR-Reasoning\OCR-Reasoning-main\outputs\Qwen3.5-9B-OCRAgentV5\T20260730_G533ef742\Qwen3.5-9B-OCRAgentV5_OCR_Reasoning_qwen-plus.xlsx"
df = pd.read_excel(XLSX)

print("=== LOG COLUMN SAMPLES (first 5, first 600 chars) ===")
for i in range(min(5, len(df))):
    log = str(df.iloc[i]['log'])[:600]
    print(f"\n--- Sample {i} (task={df.iloc[i]['task']}) ---")
    print(log)

print("\n\n=== UNIQUE LOG FIRST 150 CHARS ===")
seen = set()
for i in range(len(df)):
    log = str(df.iloc[i]['log'])[:150] if pd.notna(df.iloc[i]['log']) else 'NaN'
    if log not in seen:
        seen.add(log)
        print(f"\n--- Log type {len(seen)} ---")
        print(log)
        if len(seen) >= 25:
            break

print("\n\n=== PREDICTION SAMPLES ===")
for i in [0, 10, 50, 100, 150]:
    if i < len(df):
        pred = df.iloc[i]['prediction']
        pred_str = str(pred) if pd.notna(pred) else 'NaN'
        print(f"\n--- Pred {i} (len={len(pred_str)}, task={df.iloc[i]['task']}, RS={df.iloc[i]['reason_score']}) ---")
        print(pred_str[:400])

print("\n\n=== PREDICTION LAST 200 CHARS (samples likely truncated) ===")
long_preds = []
for i in range(len(df)):
    pred = df.iloc[i]['prediction']
    if pd.notna(pred) and len(str(pred)) > 5000:
        long_preds.append(i)

print(f"Found {len(long_preds)} predictions > 5000 chars")
for i in long_preds[:8]:
    pred_str = str(df.iloc[i]['prediction'])
    print(f"\n--- Long Pred {i} (len={len(pred_str)}) last 300 chars ---")
    print(pred_str[-300:])

print("\n\n=== RES vs ANSWER vs REASON_SCORE samples ===")
for i in range(min(20, len(df))):
    res = df.iloc[i]['res']
    ans = df.iloc[i]['answer']
    rs = df.iloc[i]['reason_score']
    task = df.iloc[i]['task']
    # Show if res == answer (simple string match)
    match = str(res).strip().lower() == str(ans).strip().lower()
    print(f"  [{i}] task={task:<35s} res={str(res)[:40]:<42s} ans={str(ans)[:40]:<42s} match={match} RS={rs}")

print("\n\n=== SAMPLES WHERE res != answer (judge mismatch vs GT) ===")
mismatch_count = 0
for i in range(len(df)):
    res = str(df.iloc[i]['res']).strip().lower() if pd.notna(df.iloc[i]['res']) else ''
    ans = str(df.iloc[i]['answer']).strip().lower() if pd.notna(df.iloc[i]['answer']) else ''
    if res and ans and res != ans:
        mismatch_count += 1
        if mismatch_count <= 10:
            print(f"  [{i}] task={df.iloc[i]['task']:<35s} res={res[:50]:<52s} ans={ans[:50]}")
print(f"  Total res != answer: {mismatch_count}/{len(df)}")
