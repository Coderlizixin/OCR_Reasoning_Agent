"""
测试：用 GT 推理步骤转成的 JSON DAG 作为 Planning，验证 Executor + Reasoning。

问题: "How many are somewhat confident that Donald Trump can manage the executive branch effectively?"

GT 推理步骤:
  step1: Identify the relevant data, locate the row 'Manage the executive branch effectively'
  step2: Examine percentages: Somewhat+Very total = 45%, Very = 21%
  step3: Calculate Somewhat = 45% - 21% = 24%

用法:
    python test_ocr_agent_v4_gt_plan.py
"""
import os
import sys
import time as _time
from PIL import Image

from test_ocr_agent_v4_planning_blind import BlindPlanningPipeline

# ============================================================================
# GT Plan: 从人工推理步骤转换的 JSON DAG
# ============================================================================
GT_PLAN = {
    "steps": [
        {
            "id": "step_1",
            "type": "ground-understand",
            "description": "Locate the row in the chart corresponding to 'Manage the executive branch effectively'",
            "depends_on": []
        },
        {
            "id": "step_2",
            "type": "ground-understand",
            "description": "Extract the combined 'Somewhat + Very' percentage value on the far right side of the bar for the 'Manage the executive branch effectively' row",
            "depends_on": ["step_1"]
        },
        {
            "id": "step_3",
            "type": "ground-understand",
            "description": "Extract the 'Very' percentage value shown inside the darkest colored segment of the bar for the 'Manage the executive branch effectively' row",
            "depends_on": ["step_1"]
        },
        {
            "id": "step_4",
            "type": "compute",
            "description": "Subtract the Very value from step_3 from the Somewhat+Very total from step_2 to calculate the Somewhat percentage",
            "depends_on": ["step_2", "step_3"]
        }
    ]
}

LOCAL_IMAGE_PATH = "test_case/test_images/image4.jpg"
QUESTION = "How many are somewhat confident that Donald Trump can manage the executive branch effectively?"


def log(msg):
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}")


def main():
    log("=" * 60)
    log("GT Plan Test: 用人工推理 DAG 驱动 Executor + Reasoning")
    log("=" * 60)

    if not os.path.exists(LOCAL_IMAGE_PATH):
        log(f"图片不存在: {LOCAL_IMAGE_PATH}")
        sys.exit(1)

    pipeline = BlindPlanningPipeline()
    orig_w, orig_h = Image.open(LOCAL_IMAGE_PATH).size

    log("\n>>> GT Plan (人工标注):")
    for s in GT_PLAN["steps"]:
        log(f"  {s['id']} ({s['type']}): deps={s['depends_on']}")
        log(f"    {s['description']}")

    # ================================================================
    # Stage 2: Executor (使用 GT Plan)
    # ================================================================
    t2 = _time.time()
    all_results, exec_status = pipeline.stage2_executor(
        GT_PLAN["steps"], LOCAL_IMAGE_PATH, orig_w, orig_h
    )
    t2_e = _time.time() - t2

    log(f"\n>>> Executor 完成 [{t2_e:.0f}s]: status={exec_status}")
    for sid, data in all_results.items():
        s = data.get('status', '?')
        r = str(data.get('result', ''))[:150]
        log(f"  {sid}: [{s}] {r}")

    # ================================================================
    # Stage 3: Reasoning (text-only)
    # ================================================================
    t3 = _time.time()
    ret, answer, resp = pipeline.stage3_reasoning(QUESTION, all_results)
    t3_e = _time.time() - t3

    # ================================================================
    # 输出
    # ================================================================
    print("\n" + "=" * 60)
    print("【最终答案】:")
    print(answer)
    print("=" * 60)

    n_ok = sum(1 for r in all_results.values() if r.get('status') == 'ok')
    n_err = len(GT_PLAN["steps"]) - n_ok
    log(f"steps={len(GT_PLAN['steps'])} ok={n_ok} err={n_err} | Exec={t2_e:.0f}s Reason={t3_e:.0f}s")


if __name__ == '__main__':
    main()
