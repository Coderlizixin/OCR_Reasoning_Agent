"""
Path B: 验证 Qwen3.5-9B 是否能用于 attention 分析实验。

核心验证点:
  1. 模型是否能加载并启用 output_attentions=True
  2. 哪些层返回注意力矩阵 (Gated DeltaNet vs Gated Attention)
  3. 图像 token 的定位方式 (special token 识别)
  4. 注意力矩阵的形状和显存开销
  5. Chat template 中 assistant 标记的识别

Qwen3.5-9B 架构 (32 layers):
  8 × (3 × Gated DeltaNet → FFN → 1 × Gated Attention → FFN)
  → 每 4 层中只有 1 层是传统 Gated Attention，其余 3 层是 DeltaNet (linear attention)
  → 可能只有 8 层返回标准 attention weights

图片和问题: 完全参考 test_attention_image_vs_text.py
"""

import torch
import os
import json
import numpy as np
from PIL import Image

# ============================================================
# Step 0: 配置 (参考 test_attention_image_vs_text.py)
# ============================================================
MODEL_PATH = "local_Qwen3.5-9B"  # 本地模型路径
IMAGE_PATH = "test_case/test_images/image26.jpg"
BBOX_JSON = "test_case/bbox_annotations/bbox_9.json"
USE_4BIT = True  # 4-bit 量化减少显存

print("=" * 70)
print("Qwen3.5-9B Attention 提取 可行性验证")
print(f"模型路径: {MODEL_PATH}")
print(f"测试图片: {IMAGE_PATH}")
print(f"测试问题: {BBOX_JSON}")
print("=" * 70)

# ============================================================
# Step 1: 检查 transformers 版本
# ============================================================
print("\n[1/7] 检查 transformers 版本...")
import transformers
print(f"  transformers version: {transformers.__version__}")
major, minor = map(int, transformers.__version__.split('.')[:2])
if major < 4 or (major == 4 and minor < 51):
    print(f"  ⚠ WARNING: transformers {transformers.__version__} may not support Qwen3.5.")
    print(f"            建议: pip install transformers>=4.51.0")
else:
    print(f"  ✅ transformers version OK")

# ============================================================
# Step 2: 检查本地模型 & tokenizer 特殊 token
# ============================================================
print("\n[2/7] 加载 processor 并检查特殊 token...")

if not os.path.exists(MODEL_PATH):
    print(f"  ❌ 模型路径不存在: {os.path.abspath(MODEL_PATH)}")
    print(f"  请确认模型已下载到该目录，或修改 MODEL_PATH 变量")
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

print(f"  Processor type: {type(processor).__name__}")
print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"  Pad token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
print(f"  EOS token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
print(f"  BOS token: {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")

# 查找与 vision/image 相关的 token
print("\n  --- 图像相关 token ---")
vision_keywords = ['vision', 'image', 'img', 'patch', 'visual']
for tok, tid in tokenizer.get_vocab().items():
    for kw in vision_keywords:
        if kw in tok.lower():
            print(f"    {tok!r} -> id={tid}")
            break

# 查找 chat template 相关 token
print("\n  --- Chat template / 对话结构 token ---")
chat_keywords = ['<|im_start|>', '<|im_end|>', 'assistant', 'user', 'system',
                 '<|vision_start|>', '<|vision_end|>', '<|image_pad|>',
                 '<|video_pad|>', '<|audio_pad|>', '<|think|>']
for tok, tid in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]):
    for kw in chat_keywords:
        if kw in tok:
            print(f"    {tok!r} -> id={tid}")
            break

# 检查 chat_template
print("\n  --- Chat template (前 600 字符) ---")
chat_tpl = getattr(tokenizer, 'chat_template', None)
if chat_tpl:
    print(f"    {chat_tpl[:600]}...")
else:
    print("    (no chat_template found)")

# 检查 image_processor
print("\n  --- Image processor ---")
img_proc = getattr(processor, 'image_processor', None)
if img_proc:
    print(f"    type: {type(img_proc).__name__}")
    for attr in ['min_pixels', 'max_pixels', 'image_mean', 'image_std', 'patch_size', 'merge_size']:
        val = getattr(img_proc, attr, None)
        if val is not None:
            print(f"    {attr}: {val}")

# ============================================================
# Step 3: 加载图片和问题 (完全参考 test_attention_image_vs_text.py)
# ============================================================
print("\n[3/7] 加载图片和问题...")

image = Image.open(IMAGE_PATH).convert('RGB')
print(f"  图片: {IMAGE_PATH} ({image.size[0]}x{image.size[1]})")

with open(BBOX_JSON, 'r', encoding='utf-8') as f:
    bbox_data = json.load(f)
question = bbox_data['question']
gt_answer = bbox_data['ground_truth']
print(f"  问题: {question[:100]}...")
print(f"  真值: {gt_answer}")

# 构造消息并查看 chat template 输出
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question}
    ]}
]

try:
    raw_chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"\n  Chat template 输出:")
    if len(raw_chat) > 800:
        print(f"    {raw_chat[:400]}")
        print(f"    ... (total {len(raw_chat)} chars) ...")
        print(f"    {raw_chat[-400:]}")
    else:
        print(f"    {raw_chat}")
except Exception as e:
    print(f"  ⚠ apply_chat_template 失败: {e}")
    # Qwen3.5 可能需要 qwen_vl_utils
    try:
        from qwen_vl_utils import process_vision_info
        images_proc, videos = process_vision_info(messages)
        raw_chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"  ✅ 使用 qwen_vl_utils 后成功")
    except ImportError:
        print(f"  ⚠ qwen_vl_utils 未安装, 尝试 pip install qwen-vl-utils")
        raise

# ============================================================
# Step 4: 加载模型
# ============================================================
print("\n[4/7] 加载模型 (4-bit 量化，可能需要几分钟)...")

gpu_available = torch.cuda.is_available()
device = "cuda" if gpu_available else "cpu"
print(f"  Device: {device}")
if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

try:
    from transformers import AutoModelForImageTextToText

    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if gpu_available else torch.float32,
        "attn_implementation": "eager",  # 必须用 eager 才能输出 attention!
    }

    if USE_4BIT and gpu_available:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print(f"  Using 4-bit quantization (BitsAndBytes)")
        except ImportError:
            print(f"  ⚠ BitsAndBytes not available, loading in bfloat16")

    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, **load_kwargs)
    model.eval()

    print(f"  ✅ 模型加载成功")
    print(f"  Model class: {type(model).__name__}")

except Exception as e:
    print(f"  ❌ AutoModelForImageTextToText 失败: {e}")
    print(f"\n  尝试 fallback: AutoModelForMultimodalLM...")
    try:
        from transformers import AutoModelForMultimodalLM
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_PATH, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.bfloat16 if gpu_available else torch.float32,
            attn_implementation="eager",
        )
        model.eval()
        print(f"  ✅ Fallback 成功")
        print(f"  Model class: {type(model).__name__}")
    except Exception as e2:
        print(f"  ❌ Fallback 也失败了: {e2}")
        raise RuntimeError("无法加载模型，请检查本地模型文件和 transformers 版本")

# 检查模型配置
print(f"\n  Model config key attributes:")
config = model.config
for attr in ['num_hidden_layers', 'hidden_size', 'num_attention_heads',
             'num_key_value_heads', 'model_type', 'architectures']:
    val = getattr(config, attr, 'N/A')
    print(f"    {attr}: {val}")

# ============================================================
# Step 5: 前向传播 + 提取 attention (仅 prompt，不 generate)
# ============================================================
print("\n[5/7] 前向传播 (output_attentions=True)...")

try:
    # Tokenize — 参考 test_attention_image_vs_text.py 的方式
    inputs = processor(
        text=[raw_chat],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    try:
        inputs = inputs.to(model.device)
    except:
        inputs = {k: v.to('cuda' if gpu_available else 'cpu') for k, v in inputs.items()}

    seq_len = inputs["input_ids"].shape[1]
    print(f"  输入序列长度: {seq_len} tokens")

    # 打印所有 input keys
    print(f"  Input keys: {list(inputs.keys())}")

    # 如果有 image_grid_thw，打印
    if "image_grid_thw" in inputs:
        grid_thw = inputs["image_grid_thw"][0]
        print(f"  image_grid_thw: {grid_thw.tolist()}")
    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        print(f"  pixel_values shape: {pv.shape}")

    # 前向传播
    print(f"\n  正在前向传播...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 检查 attention 输出
    attentions = outputs.attentions

    if attentions is None:
        print(f"\n  ❌ output_attentions=True 但 attentions 为 None!")
        print(f"  该模型可能不支持返回注意力权重")
        print(f"  可能原因: Gated DeltaNet 层不产生标准 attention matrix")
    else:
        print(f"\n  ✅ 成功提取注意力矩阵!")
        num_layers = len(attentions)
        print(f"  Attention tuple length: {num_layers}")

        # 逐层检查
        print(f"\n  {'Layer':<8} {'Shape':<28} {'Status':<16} {'All Zero':<12} {'Mean±Std'}")
        print(f"  {'-'*85}")

        layers_with_attn = []
        layers_without_attn = []

        for l in range(num_layers):
            if attentions[l] is None:
                layers_without_attn.append(l)
                print(f"  L{l:<7} {'None':<28} {'❌ NONE':<16} {'N/A':<12} {'N/A'}")
                continue

            attn_l = attentions[l]
            if isinstance(attn_l, tuple):
                first_head = attn_l[0]
                shape_str = f"tuple[{len(attn_l)}] × {list(first_head.shape)}"
                avg_attn = torch.stack(list(attn_l), dim=0).mean(dim=0)
            elif isinstance(attn_l, torch.Tensor):
                shape_str = str(list(attn_l.shape))
                avg_attn = attn_l.mean(dim=0) if attn_l.dim() >= 4 else attn_l
            else:
                shape_str = f"type={type(attn_l).__name__}"
                layers_without_attn.append(l)
                print(f"  L{l:<7} {shape_str:<28} {'❌ UNKNOWN':<16}")
                continue

            # 检查是否为全零矩阵
            if avg_attn.dim() < 2:
                is_zero = True
            else:
                is_zero = bool(torch.allclose(avg_attn, torch.zeros_like(avg_attn), atol=1e-8))

            mean_val = float(avg_attn.mean()) if not is_zero else 0.0
            std_val = float(avg_attn.std()) if not is_zero else 0.0

            status = "✅ GATED ATTENTION" if not is_zero else "⚠ zero matrix"
            zero_str = "YES" if is_zero else "NO"

            print(f"  L{l:<7} {shape_str:<28} {status:<16} {zero_str:<12} {mean_val:.6f} ± {std_val:.6f}")

            if not is_zero:
                layers_with_attn.append(l)
            else:
                layers_without_attn.append(l)

        print(f"\n  --- 汇总 ---")
        print(f"  返回非空注意力的层: {len(layers_with_attn)}/{num_layers}")
        print(f"  有实际注意力值的层 (Gated Attention): {layers_with_attn}")
        print(f"  无注意力值/为零的层 (Gated DeltaNet): {layers_without_attn}")

        # 验证架构假设: 每 4 层有 1 层 Gated Attention
        if layers_with_attn:
            expected_pattern = set(range(3, num_layers, 4))  # L3, L7, L11, L15, L19, L23, L27, L31
            actual_set = set(layers_with_attn)
            match = expected_pattern == actual_set
            print(f"\n  架构验证:")
            print(f"    预期 Gated Attention 层: {sorted(expected_pattern)}")
            print(f"    实际 Gated Attention 层: {sorted(actual_set)}")
            print(f"    {'✅ 匹配! 每 4 层中的第 4 层是 Gated Attention' if match else '⚠ 与预期不完全一致'}")

        if len(layers_with_attn) == 0:
            print(f"\n  ❌ 没有层返回有效的注意力矩阵!")
            print(f"  该实验无法直接使用 Qwen3.5-9B")
        elif len(layers_with_attn) < num_layers:
            print(f"\n  ⚠ 只有 {len(layers_with_attn)}/{num_layers} 层有标准注意力")
            print(f"  这是预期的 — Gated DeltaNet 层使用 linear attention")
            print(f"  实验可调整为: 只分析这 {len(layers_with_attn)} 个 Gated Attention 层的注意力")

        # 展示示例层的详细信息
        if layers_with_attn:
            example_layer = layers_with_attn[0]
            attn_l = attentions[example_layer]
            if isinstance(attn_l, tuple):
                example_tensor = attn_l[0]
            else:
                example_tensor = attn_l

            print(f"\n  --- 示例注意力矩阵 (L{example_layer}, 首个 head) ---")
            print(f"  Shape: {list(example_tensor.shape)}")
            print(f"  含义: [batch={example_tensor.shape[0]}, heads={example_tensor.shape[1]}, "
                  f"Q_seq={example_tensor.shape[2]}, K_seq={example_tensor.shape[3]}]")

except Exception as e:
    print(f"\n  ❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================================
# Step 6: Token 分组识别验证
# ============================================================
print("\n[6/7] Token 分组识别验证...")

input_ids = inputs["input_ids"][0].cpu().tolist()
tokens = [tokenizer.decode([tok]) for tok in input_ids]

# 查找所有结构 token
vision_start_positions = []
vision_end_positions = []
image_pad_positions = []
im_start_positions = []
im_end_positions = []
assistant_positions = []

for i, tok in enumerate(tokens):
    if 'vision_start' in tok:
        vision_start_positions.append(i)
    elif 'vision_end' in tok:
        vision_end_positions.append(i)
    elif 'image_pad' in tok:
        image_pad_positions.append(i)
    elif 'im_start' in tok:
        im_start_positions.append(i)
    elif 'im_end' in tok:
        im_end_positions.append(i)
    elif 'assistant' in tok.lower() and '<|' in tok:
        assistant_positions.append(i)

print(f"\n  结构 token 统计:")
print(f"    vision_start: {len(vision_start_positions)} 个, 位置={vision_start_positions}")
print(f"    vision_end:   {len(vision_end_positions)} 个, 位置={vision_end_positions}")
print(f"    image_pad:    {len(image_pad_positions)} 个, 范围=[{min(image_pad_positions) if image_pad_positions else 'N/A'}, {max(image_pad_positions) if image_pad_positions else 'N/A'}]")  # noqa: E501
print(f"    im_start:     {len(im_start_positions)} 个, 位置={im_start_positions}")
print(f"    im_end:       {len(im_end_positions)} 个, 位置={im_end_positions}")
print(f"    assistant:    {len(assistant_positions)} 个, 位置={assistant_positions}")

# 打印序列中所有特殊 token
print(f"\n  --- 序列中所有特殊 token ---")
special_in_seq = []
for i, tok in enumerate(tokens):
    if tok.startswith('<|') or (tok.startswith('<') and tok.endswith('>')):
        special_in_seq.append((i, tok))
print(f"  共 {len(special_in_seq)} 个特殊 token:")
for i, tok in special_in_seq[:60]:
    print(f"    [{i:5d}] {tok}")
if len(special_in_seq) > 60:
    print(f"    ... 还有 {len(special_in_seq) - 60} 个")

# 确定图像 patch token 范围
image_indices = []
if vision_start_positions and vision_end_positions:
    vs = vision_start_positions[0]
    ve = vision_end_positions[0]
    image_indices = list(range(vs + 1, ve))
    print(f"\n  ✅ 图像 token: [{vs+1}, {ve-1}] 共 {len(image_indices)} 个")
    # 验证：这些 token 是否都是 image_pad
    pad_count = sum(1 for i in image_indices if 'image_pad' in tokens[i])
    print(f"     其中 image_pad token: {pad_count}/{len(image_indices)}")
elif image_pad_positions:
    # 找连续的 image_pad 区间
    image_indices = image_pad_positions
    print(f"\n  ⚠ 未找到 vision_start/end, 使用 {len(image_indices)} 个 image_pad 作为图像 token")
else:
    print(f"\n  ❌ 无法定位图像 token!")

# 确定 assistant / 输出起始位置
output_start = None
if assistant_positions:
    output_start = assistant_positions[0]
    print(f"  ✅ Assistant 起始: position {output_start}")
elif im_start_positions:
    for pos in im_start_positions:
        context = tokenizer.decode(input_ids[pos:pos+5])
        if 'assistant' in context.lower():
            output_start = pos
            print(f"  ✅ Assistant im_start: position {output_start}")
            break
    if output_start is None:
        print(f"  ⚠ 找到 im_start={im_start_positions} 但无法确定哪个是 assistant")
else:
    # 尝试从 chat_template 输出中找 'assistant' 文本
    for i, tok in enumerate(tokens):
        if 'assistant' in tok.lower():
            output_start = i
            print(f"  ✅ 从 token 文本中找到 assistant: position {i}")
            break

if output_start is not None:
    output_indices = list(range(output_start, len(input_ids)))
    print(f"     输出 token 数量 (prompt 中): {len(output_indices)}")
else:
    print(f"  ⚠ 未能确定 assistant/输出起始位置")

# 文本 token (非图像)
all_set = set(range(len(input_ids)))
text_indices_all = sorted(list(all_set - set(image_indices)))

print(f"\n  --- Token 分组总结 ---")
print(f"  总 token 数:          {len(input_ids)}")
print(f"  图像 patch token:     {len(image_indices)}")
print(f"  文本 token (其他):    {len(text_indices_all)}")
if output_start is not None:
    print(f"  输出起始位置:         {output_start}")

# ============================================================
# Step 7: 显存和计算量估算
# ============================================================
print("\n[7/7] 显存和计算量估算...")

if gpu_available:
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  GPU 显存已分配: {mem_allocated:.2f} GB")
    print(f"  GPU 显存已预留: {mem_reserved:.2f} GB")
    print(f"  GPU 显存空闲:   {torch.cuda.get_device_properties(0).total_memory/1024**3 - mem_reserved:.2f} GB")

    if 'layers_with_attn' in dir() and layers_with_attn:
        example_layer = layers_with_attn[0]
        attn_l = attentions[example_layer]
        if isinstance(attn_l, tuple):
            example_tensor = attn_l[0]
        else:
            example_tensor = attn_l
        single_layer_size = example_tensor.element_size() * example_tensor.numel()
        total_attn_size = single_layer_size * len(layers_with_attn)

        print(f"\n  Attention 矩阵存储估算 (仅 prompt):")
        print(f"    单层 attention:  {single_layer_size / 1024**2:.2f} MB")
        print(f"    有注意力的层数:  {len(layers_with_attn)}")
        print(f"    总 attention:    {total_attn_size / 1024**3:.2f} GB")

        # 完整实验估算 (生成 1024 token)
        est_full = seq_len + 1024
        scale = (est_full / seq_len) ** 2
        print(f"\n  完整实验估算 (生成 1024 token, 总序列 ~{est_full}):")
        print(f"    序列膨胀: {seq_len} → {est_full} (×{est_full/seq_len:.1f})")
        print(f"    Attention 膨胀: ×{scale:.1f} (O(n²))")
        print(f"    预估总 attention 存储: {total_attn_size * scale / 1024**3:.2f} GB")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("验证总结")
print("=" * 70)

checks = []
checks.append(("模型加载", True,
               f"{type(model).__name__} from {os.path.abspath(MODEL_PATH)}"))

attn_ok = 'attentions' in dir() and attentions is not None and len(layers_with_attn) > 0
checks.append(("Attention 提取", attn_ok,
               f"{len(layers_with_attn)}/{num_layers} 层" if attn_ok else "无有效注意力矩阵"))

img_token_ok = len(image_indices) > 0
checks.append(("图像 token 识别", img_token_ok,
               f"{len(image_indices)} 个图像 patch token"))

assistant_ok = output_start is not None
checks.append(("Assistant 标记", assistant_ok,
               f"位置={output_start}" if assistant_ok else "未找到"))

for name, ok, detail in checks:
    print(f"  {'✅' if ok else '❌'} {name:<22}: {detail}")

all_ok = all(c[1] for c in checks)

if all_ok:
    print(f"\n  ✅ 所有检查通过! Qwen3.5-9B 可用于 attention 分析实验")
    print(f"\n  ⚠ 注意事项 (与 GLM-4.6V 的差异):")
    print(f"    1. 只有 {len(layers_with_attn)}/{num_layers} 层有传统注意力 (Gated Attention 层)")
    print(f"       Gated DeltaNet 层返回 None 或全零矩阵")
    print(f"    2. analyze_attention() 需改为只遍历有注意力的层")
    print(f"    3. identify_token_groups() 需改为 Qwen 的 token 结构")
    print(f"       - GLM: <|begin_of_image|>...<|end_of_image|> + <|assistant|>")
    print(f"       - Qwen3.5: <|vision_start|>...<|vision_end|> (内含 <|image_pad|>)")
    print(f"       - Qwen3.5 的 assistant 标记是 <|im_start|>assistant")
    print(f"    4. Chat template 格式不同，需用 processor.apply_chat_template()")
    print(f"    5. image_grid_thw 可能格式不同，影响 patch 热力图绘制")
else:
    print(f"\n  ⚠ 部分检查未通过:")
    for name, ok, detail in checks:
        if not ok:
            print(f"    ❌ {name}: {detail}")

print("\n完成!")
