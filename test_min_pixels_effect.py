"""
验证 extra_body 的 shortest_edge 是否被服务端生效。
用同一张图 + 同一段文本，切换两个不同的 shortest_edge，
对比 prompt_tokens：差值即为图像被放大带来的额外 visual token。
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from vlmeval.api.gpt import GPT4V

# 测试图片路径
image_path = "test_case/test_images/image0.jpg"

# 同一段文本提示
text_prompt = "Describe this image briefly."

test_configs = [
    ("512*32*32", {"shortest_edge": 512 * 32 * 32, "longest_edge": 8192 * 32 * 32}),
    ("4096*32*32", {"shortest_edge": 4096 * 32 * 32, "longest_edge": 8192 * 32 * 32}),
]

results = {}
for label, size_cfg in test_configs:
    print(f"\n{'='*60}")
    print(f"测试: shortest_edge={label} ({size_cfg['shortest_edge']} px)")
    print(f"{'='*60}")

    model = GPT4V(
        model="Qwen3.5-9B",
        api_base="http://10.3.112.34:22240/v1/chat/completions",
        key="sk-EMPTY",
        temperature=0.0,
        img_size=-1,
        img_detail="high",
        extra_body={
            "mm_processor_kwargs": {
                "images_kwargs": {
                    "size": size_cfg,
                }
            }
        },
    )

    inputs = [
        {"type": "text", "value": text_prompt},
        {"type": "image", "value": image_path},
    ]

    ret_code, answer, resp = model.generate_inner(inputs)
    resp_json = resp.json() if hasattr(resp, 'json') else {}
    usage = resp_json.get("usage", {})

    prompt_tokens = usage.get("prompt_tokens", "N/A")
    completion_tokens = usage.get("completion_tokens", "N/A")
    total_tokens = usage.get("total_tokens", "N/A")

    results[label] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    print(f"prompt_tokens: {prompt_tokens}")
    print(f"completion_tokens: {completion_tokens}")
    print(f"answer preview: {answer[:200]}")

# 汇总对比
print(f"\n{'='*60}")
print("对比汇总")
print(f"{'='*60}")
print(f"{'shortest_edge':<20} {'prompt_tokens':>14} {'completion_tokens':>18}")
print("-" * 52)
for label, r in results.items():
    print(f"{label:<20} {str(r['prompt_tokens']):>14} {str(r['completion_tokens']):>18}")

if all(isinstance(r["prompt_tokens"], int) for r in results.values()):
    vals = list(results.values())
    diff = vals[1]["prompt_tokens"] - vals[0]["prompt_tokens"]
    print(f"\n>>> prompt_tokens 差值: {diff}")
    if diff > 100:
        print(f">>> 结论: shortest_edge 生效！高分辨率产生了额外 {diff} 个 visual token")
    else:
        print(f">>> 结论: shortest_edge 可能未生效，差值太小")
else:
    print("\n>>> 服务端未返回 usage，无法对比（vLLM 需开启 token 统计）")
