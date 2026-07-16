"""
独立测试脚本：使用 planning 伪标签测试 V4 流水线。
与 test_ocr_agent_v4.py 一致，但 Stage 1 (Planning) 不调用模型，
而是从 test_case/planning_pseudo_labels/ 加载由 reasoning GT 构建的伪标签。

用法:
    python test_ocr_agent_v4_pseudo_plan.py
"""
import os
import sys
import json
import time as _time

# 测试配置 — 使用 dataset index 21（有匹配的 pseudo-label）
# Pseudo-label: 4 steps (2 ground-understand + 2 compute)
# GT answer: "20"
LOCAL_IMAGE_PATH = "test_case/test_images/dataset_idx21.png"
QUESTION = "How many days does this campaign last?"


def log(msg):
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}")


def main():
    log("=" * 60)
    log("OCRAgentV4PseudoPlan 独立测试 (使用 GT Planning 伪标签)")
    log("=" * 60)

    # 检查图片是否存在
    if not os.path.exists(LOCAL_IMAGE_PATH):
        log(f"测试图片不存在: {LOCAL_IMAGE_PATH}")
        log("请先准备测试图片，或修改 LOCAL_IMAGE_PATH")
        sys.exit(1)

    log(f"图片: {LOCAL_IMAGE_PATH}")
    log(f"问题: {QUESTION}")

    # 导入并初始化 agent
    from vlmeval.vlm.ocr_agent_v4_pseudo_plan import OCRAgentV4PseudoPlan

    agent = OCRAgentV4PseudoPlan(
        model='Qwen3.5-9B',
        api_base='http://10.3.112.34:22630/v1/chat/completions',
        key='sk-EMPTY',
        temperature=0.0,
        temp_planning=0.0,
        temp_locate=0.1,
        temp_understand=0.2,
        temp_compute=0.1,
        temp_logical=0.1,
        temp_reasoning=0.2,
        max_tokens=6144,
        img_size=-1,
        img_detail='high',
        verbose=True,
        retry=3,
        wait=3,
        pseudo_plan_dir="test_case/planning_pseudo_labels",
        trace_dir="test_case/v4_pseudo_plan_traces",
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

    log("Agent 初始化完成")
    log(f"API Base: {agent.api_base}")
    log(f"Model: {agent.model}")
    log(f"Pseudo Plan Dir: {agent.pseudo_plan_dir}")
    log(f"Loaded {len(agent._pseudo_plans_by_idx)} plans by index, "
        f"{len(agent._pseudo_plans_by_q)} unique questions")

    # 构造输入
    inputs = [
        {'type': 'text', 'value': QUESTION},
        {'type': 'image', 'value': LOCAL_IMAGE_PATH},
    ]

    # 运行推理
    log("开始运行 V4 Pseudo-Plan Pipeline...")
    t_start = _time.time()

    ret, answer, resp = agent.generate_inner(inputs)

    t_total = _time.time() - t_start

    log(f"\n运行完成! 耗时: {t_total:.0f}s")
    log(f"返回码: {ret}")
    log(f"答案长度: {len(answer) if answer else 0} chars")

    print("\n" + "=" * 60)
    print("【最终答案】:")
    print(answer)
    print("=" * 60)

    # 打印执行追踪
    if agent.last_trace:
        print("\n" + "=" * 60)
        print("【执行追踪 (Execution Trace)】:")
        print("=" * 60)
        trace = agent.last_trace
        print(f"  整体状态: {trace['overall_status']}")
        print(f"  总耗时:   {trace['total_time_s']}s")
        print(f"\n  Planning:")
        p = trace['planning']
        print(f"    来源: pseudo-label (GT), 状态: {p['status']}, 步骤数: {p['n_steps']}")
        if p['steps']:
            for s in p['steps']:
                print(f"      {s['id']} ({s['type']}): deps={s.get('depends_on', [])}")
                print(f"        {s['description'][:120]}")
        print(f"\n  Executor:")
        e = trace['executor']
        print(f"    状态: {e['overall_status']}, ok={e['n_ok']}/{e['n_total']}, 耗时: {e['time_s']}s")
        for sid, sd in e['steps'].items():
            status = sd['status']
            marker = '[OK]' if status == 'ok' else '[ERR]'
            retry = f' (retry={sd.get("retry_count", 0)})' if sd.get('retry_count', 0) > 0 else ''
            if status == 'ok':
                print(f"      {marker} {sid} ({sd['type']}){retry}: {str(sd.get('result', ''))[:120]}")
            else:
                phase = sd.get('failed_phase', '')
                phase_tag = f' [{phase}]' if phase else ''
                print(f"      {marker} {sid} ({sd['type']}){phase_tag}{retry}: {sd.get('error', '')[:150]}")
        print(f"\n  Reasoning:")
        r = trace['reasoning']
        print(f"    模式: {r['mode']}, 有答案: {r['has_answer']}, 耗时: {r['time_s']}s")
        print(f"    答案预览: {str(r.get('final_answer', ''))[:200]}")

    # 保存结果
    output_dir = "test_case/v4_pseudo_plan_test_output"
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "image_path": LOCAL_IMAGE_PATH,
        "question": QUESTION,
        "ret_code": ret,
        "answer": answer,
        "total_time_s": round(t_total, 1),
    }
    result_path = os.path.join(output_dir, "test_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"结果已保存到: {result_path}")


if __name__ == '__main__':
    main()
