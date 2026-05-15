import torch
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from transformers import AutoProcessor, Glm4vForConditionalGeneration


def get_color_based_background_mask(image_pil, vis_h, vis_w):
    """
    通过 HSV 色彩空间，物理提取图表的纯背景 Patch。
    返回一个 (vis_h, vis_w) 的二维二值掩码 (1=背景需擦除, 0=保留的图文)
    """
    # 1. 转换为 OpenCV 格式
    image_np = np.array(image_pil.convert('RGB'), dtype=np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    # 2. 转换到 HSV 空间
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]  # 饱和度 (0-255)
    v = hsv[:, :, 2]  # 明度 (0-255)

    # 3. 像素级背景提取：低饱和度 (发白/发灰) 且 高明度 (不包含纯黑文字)
    bg_pixel_mask = ((s < 35) & (v > 200)).astype(np.uint8)

    # 4. 形态学操作：轻微膨胀文字和色块，防止边缘被误判为背景
    kernel = np.ones((3, 3), np.uint8)
    fg_pixel_mask = 1 - bg_pixel_mask
    fg_pixel_mask = cv2.dilate(fg_pixel_mask, kernel, iterations=1)
    bg_pixel_mask = 1 - fg_pixel_mask

    # 5. Patch 级网格化 (Pooling)
    mask_2d = np.zeros((vis_h, vis_w), dtype=np.uint8)
    patch_h = h / vis_h
    patch_w = w / vis_w

    for i in range(vis_h):
        for j in range(vis_w):
            y1, y2 = int(i * patch_h), int((i + 1) * patch_h)
            x1, x2 = int(j * patch_w), int((j + 1) * patch_w)

            patch_pixels = bg_pixel_mask[y1:y2, x1:x2]
            bg_ratio = np.sum(patch_pixels) / patch_pixels.size

            # 只要该方块内 90% 以上是背景色，就判定为无关 Token
            if bg_ratio > 0.90:
                mask_2d[i, j] = 1
            else:
                mask_2d[i, j] = 0

    return mask_2d


def apply_zoomtext_and_glc(model, prompt, inputs, processor, image, fusion_weight=0.1, glimpse_relax_factor=0.2):
    # =========================================================
    # 🌟 新增：计算通用描述指令 (q') 的底噪 Attention
    # =========================================================
    messages_prime = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Write a general description of the image."}
            ]
        }
    ]
    prompt_prime = processor.apply_chat_template(messages_prime, tokenize=False, add_generation_prompt=True)
    inputs_prime = processor(text=prompt_prime, images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_prime = model(**inputs_prime, output_attentions=True)
        attentions_prime = out_prime.attentions

    # =========================================================
    # 提取真实问题 (q) 的 Attention
    # =========================================================
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
        attentions = out.attentions

    num_layers = len(attentions)
    input_ids = inputs["input_ids"][0].cpu().tolist()

    # ---------------------------------------------------------
    # 步骤 A：精准定位 Image Token 和 Query Token 的索引边界
    # ---------------------------------------------------------
    image_start, image_end = -1, -1
    tokens = [processor.decode([tok]) for tok in input_ids]

    for i, token in enumerate(tokens):
        if token == '<|begin_of_image|>':
            image_start = i + 1
        elif token == '<|end_of_image|>':
            image_end = i
            break

    if image_start == -1 or image_end == -1:
        image_start, image_end = 5, 1205

    image_indices = list(range(image_start, image_end))

    # 🌟 核心修正：只取 Prompt 的最后一个 Token（Starting Answer Token）
    query_idx_q = [inputs["input_ids"].shape[1] - 1]
    query_idx_prime = [inputs_prime["input_ids"].shape[1] - 1]

    # ---------------------------------------------------------
    # 步骤 B：弃用注意力筛选，改用 HSV 物理提取
    # ---------------------------------------------------------
    grid_thw = inputs["image_grid_thw"][0]
    orig_h, orig_w = grid_thw[1].item(), grid_thw[2].item()
    vis_h, vis_w = orig_h // 2, orig_w // 2

    print(f"[HSV Mask] 正在通过色彩空间提取背景... (网格: {vis_h}x{vis_w})")
    mask_2d = get_color_based_background_mask(image, vis_h, vis_w)
    mask_1d = mask_2d.flatten()

    text_token_indices = []
    irrelevant_img_indices = []

    for local_idx, is_bg in enumerate(mask_1d):
        global_idx = image_start + local_idx
        if is_bg == 1:
            irrelevant_img_indices.append(global_idx)
        else:
            text_token_indices.append(global_idx)

    if len(text_token_indices) == 0:
        text_token_indices = image_indices

    print(
        f"[HSV Mask] 精准锁定 {len(text_token_indices)} 个高优图文 Patch，识别出 {len(irrelevant_img_indices)} 个背景 Patch！")

    # ---------------------------------------------------------
    # 步骤 C：寻找最佳视觉锚定层 l* -> (现在使用 Relative Attention 寻找峰值)
    # ---------------------------------------------------------
    layer_scores = []
    epsilon = 1e-6

    # 将全局索引转为局部图像索引，方便切片计算
    local_text_indices = [idx - image_start for idx in text_token_indices]

    for l in range(num_layers):
        A_l = attentions[l][0].mean(dim=0).cpu().float().numpy()
        A_l_prime = attentions_prime[l][0].mean(dim=0).cpu().float().numpy()

        q_to_img = A_l[query_idx_q][:, image_indices].mean(axis=0)
        q_prime_to_img = A_l_prime[query_idx_prime][:, image_indices].mean(axis=0)

        # 相对注意力计算
        rel_attn = q_to_img / (q_prime_to_img + epsilon)

        # l* 为对文字区域相对注意力最高的那一层
        target_score = rel_attn[local_text_indices].mean()
        layer_scores.append(target_score)

    best_layer_idx = np.argmax(layer_scores[5:]) + 5  # 避开前5层的极端噪音
    print(
        f"根据 Relative Attention 评估，视觉锚定最强层(l*)为 Layer {best_layer_idx} (Score: {layer_scores[best_layer_idx]:.4f})")

    # 绘制trend图 (传入两次注意力结果)
    plot_attention_trend_across_layers(
        attentions,
        attentions_prime,
        query_idx_q,
        query_idx_prime,
        image_indices,
        text_token_indices,
        irrelevant_img_indices
    )

    print("[Visualization] 正在将底层 Token 索引逆向映射为二维掩码...")

    # ---------- 下面的代码与你原来完全一样（掩码涂灰、挂载 GLC Hook、生成答案） ----------
    image_np = np.array(image.convert('RGB'), dtype=np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    mask_resized = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = image_bgr.copy()
    overlay[mask_resized == 1] = [0, 0, 255]
    alpha = 0.5
    combined_img = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    save_path = 'test_case/visualization/irrelevant_mask_overlay.jpg'
    cv2.imwrite(save_path, combined_img)

    clean_image_np = image_np.copy()
    clean_image_np[mask_resized == 1] = [127, 127, 127]
    clean_pil_image = Image.fromarray(clean_image_np)
    clean_pil_image.save('clean_input_image.jpg')

    clean_inputs = processor(text=prompt, images=[clean_pil_image], return_tensors="pt").to(model.device)

    inputs['pixel_values'] = clean_inputs['pixel_values']
    if 'image_grid_thw' in clean_inputs:
        inputs['image_grid_thw'] = clean_inputs['image_grid_thw']

    saved_best_layer_hidden = None
    hooks = []

    def hook_save_best_layer(module, hook_inputs, hook_outputs):
        nonlocal saved_best_layer_hidden
        if isinstance(hook_outputs, tuple):
            saved_best_layer_hidden = hook_outputs[0].clone()
        else:
            saved_best_layer_hidden = hook_outputs.clone()
        return hook_outputs

    def hook_fuse_final_layer(module, hook_inputs, hook_outputs):
        nonlocal saved_best_layer_hidden
        if saved_best_layer_hidden is not None:
            w = fusion_weight
            if isinstance(hook_outputs, tuple):
                original_hidden = hook_outputs[0]
                fused_hidden = (1 - w) * original_hidden + w * saved_best_layer_hidden
                return (fused_hidden,) + hook_outputs[1:]
            else:
                original_hidden = hook_outputs
                fused_hidden = (1 - w) * original_hidden + w * saved_best_layer_hidden
                return fused_hidden
        return hook_outputs

    transformer_layers = None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 10:
            name_lower = name.lower()
            if not any(k in name_lower for k in ['vision', 'visual', 'vit', 'vpm']):
                transformer_layers = module
                break

    if transformer_layers is None:
        raise ValueError("动态查找失败！无法在当前模型架构中定位到语言模型的 Transformer Layers！")

    final_layer_idx = len(transformer_layers) - 1

    h1 = transformer_layers[best_layer_idx].register_forward_hook(hook_save_best_layer)
    h2 = transformer_layers[final_layer_idx].register_forward_hook(hook_fuse_final_layer)
    hooks.extend([h1, h2])

    print(f"[GLC] 开始携带纠偏特征进行生成 (Fusion Weight={fusion_weight})...")
    with torch.no_grad():
        generate_kwargs = {"max_new_tokens": 2048, "do_sample": False, "temperature": 0.0}
        generated_ids = model.generate(**inputs, **generate_kwargs)

    for h in hooks:
        h.remove()

    return generated_ids


def plot_attention_trend_across_layers(attentions, attentions_prime, query_idx_q, query_idx_prime, image_indices,
                                       text_token_indices, irrelevant_img_indices):
    """
    绘制 Relative Attention (A_q / A_q') 的层级变化趋势
    """
    print("[Visualization] 正在计算并绘制每一层的 Relative Attention 变化趋势...")
    num_layers = len(attentions)

    target_focus_scores = []
    bg_focus_scores = []
    epsilon = 1e-6

    # 映射回相对于图像块起点的局部索引 (0 到 num_img_tokens-1)
    image_start = image_indices[0]
    local_text_indices = [idx - image_start for idx in text_token_indices]
    local_bg_indices = [idx - image_start for idx in irrelevant_img_indices]

    for l in range(num_layers):
        # 第 l 层：真实问题的注意力 (q)
        layer_attn = attentions[l][0].mean(dim=0).cpu().float().numpy()
        q_to_img = layer_attn[query_idx_q][:, image_indices].mean(axis=0)

        # 第 l 层：通用问题的底噪注意力 (q')
        layer_attn_prime = attentions_prime[l][0].mean(dim=0).cpu().float().numpy()
        q_prime_to_img = layer_attn_prime[query_idx_prime][:, image_indices].mean(axis=0)

        # 核心：计算 Relative Attention
        rel_attn = q_to_img / (q_prime_to_img + epsilon)

        # 分离出目标文字和背景的平均相对分数
        target_score = rel_attn[local_text_indices].mean()
        bg_score = rel_attn[local_bg_indices].mean()

        target_focus_scores.append(target_score)
        bg_focus_scores.append(bg_score)

    # ================= 绘制趋势图 =================
    plt.figure(figsize=(12, 6))

    plt.plot(range(num_layers), target_focus_scores,
             label="Relative Focus on Relevant Text (Signal > 1.0)",
             color="#d62728", linewidth=2.5, marker='o', markersize=4)

    plt.plot(range(num_layers), bg_focus_scores,
             label="Relative Focus on Background",
             color="#1f77b4", linewidth=2, linestyle='--')

    # 绘制 1.0 的基准线
    plt.axhline(y=1.0, color='green', linestyle='-', alpha=0.3, label="Baseline (A_q = A_q')")

    # 标出 l* (视觉锚定最强层)
    best_layer = np.argmax(target_focus_scores[5:]) + 5  # 从第5层后开始找峰值
    plt.axvline(x=best_layer, color='gray', linestyle=':', label=f"Best Layer (l* = {best_layer})")
    plt.text(best_layer + 0.5, target_focus_scores[best_layer], 'Peak Visual Anchoring', color='black')

    plt.title("Relative Attention Trend (A_q / A_q') Across Transformer Layers", fontsize=14, fontweight='bold')
    plt.xlabel("Transformer Layer Depth", fontsize=12)
    plt.ylabel("Relative Attention Score", fontsize=12)
    plt.xticks(range(0, num_layers, 2))
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    save_path = "test_case/visualization/layerwise_relative_attention_trend.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Visualization] Relative趋势图已保存至: {os.path.abspath(save_path)}")

if __name__ == '__main__':
    # 1. 加载模型（千万别忘掉 eager）
    model_path = "local_GLM-4.6V-Flash"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading GLM-4.1V-Thinking for VISUAL SALIENCY Map...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # 2. 构造干净的 OCR 推理 Prompt（保持英文）
    image_path = "test_case/image0.jpg"
    # image_path = "image2.jpg"
    image = Image.open(image_path).convert('RGB')

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # 这里只需放一个类型占位符
                {"type": "text",
                 "text": "Please pay close attention to the textual information in the image, as well as the key elements "
                         "specified in the question, such as the objects, relationships, and constraints. "
                         "Then answer the following question within the tags <q> and </q> step-by-step "
                         "according to the textual content, explaining your reasoning process clearly at each stage and the "
                         "text clues you use. After presenting your reasoning, provide the final simple answer at last, "
                         "making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. "
                         "Question: <q>What is the average number of the genres of movies made by directors who have worked in drama genres?</q>"
                         "Answer"}
            ]
        }
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    # 3. 前向传播：生成并提取 Full Attention
    with torch.no_grad():
        generated_ids = apply_zoomtext_and_glc(model, prompt, inputs, processor, image)
        # answer = processor.decode(
        #     generated_ids[0][inputs["input_ids"].shape[1]:],
        #     skip_special_tokens=True
        # )
