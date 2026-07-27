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


def get_gt_region_mask(bbox_json_path, vis_h, vis_w):
    """
    根据 bbox 标注文件生成 GT 区域掩码。
    坐标规范：千分化归一 [0, 1000]，(0,0)=top-left, (1000,1000)=bottom-right。
    返回一个 (vis_h, vis_w) 的二维二值掩码 (1=GT区域, 0=非GT区域)。
    """
    import json

    with open(bbox_json_path, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)

    bboxes = bbox_data['bboxes_stage2_refined']

    mask_2d = np.zeros((vis_h, vis_w), dtype=np.uint8)

    for bbox_item in bboxes:
        bbox = bbox_item['bbox_norm']  # [xmin, ymin, xmax, ymax] in [0, 1000]
        xmin, ymin, xmax, ymax = bbox

        # 千分化归一 → patch 网格坐标
        # pixel = norm / 1000 * image_size
        # patch_idx = pixel / image_size * grid_size = norm / 1000 * grid_size
        i_min = int(ymin / 1000.0 * vis_h)
        i_max = int(np.ceil(ymax / 1000.0 * vis_h))
        j_min = int(xmin / 1000.0 * vis_w)
        j_max = int(np.ceil(xmax / 1000.0 * vis_w))

        # 裁剪到有效范围
        i_min = max(0, i_min)
        i_max = min(vis_h, i_max)
        j_min = max(0, j_min)
        j_max = min(vis_w, j_max)

        mask_2d[i_min:i_max, j_min:j_max] = 1

    return mask_2d


def apply_zoomtext_and_glc(model, prompt, inputs, processor, image,
                           fusion_weight=0.1, glimpse_relax_factor=0.2,
                           bbox_json_path=None):
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

    # 🌟 改为取整个 question（不包括图像区域）对应的所有 tokens
    # query_idx_q: image_end 之后的所有文本 token（问题文本 + 生成提示）
    query_idx_q = list(range(image_end + 1, inputs["input_ids"].shape[1]))
    # query_idx_prime: 通用描述提示的文本 token 部分（同样不包括图像）
    query_idx_prime = list(range(image_end + 1, inputs_prime["input_ids"].shape[1]))

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
    # 🌟 步骤 B2：加载 GT 标注，区分"真正需要看"的区域
    # ---------------------------------------------------------
    gt_token_indices = []
    non_gt_text_indices = []

    if bbox_json_path is not None and os.path.exists(bbox_json_path):
        print(f"[GT Mask] 正在从 {bbox_json_path} 加载 GT 区域标注...")
        gt_mask_2d = get_gt_region_mask(bbox_json_path, vis_h, vis_w)
        gt_mask_1d = gt_mask_2d.flatten()

        for local_idx, is_gt in enumerate(gt_mask_1d):
            global_idx = image_start + local_idx
            if is_gt == 1:
                gt_token_indices.append(global_idx)

        # 非 GT 文字区域 = HSV 判定为文字 但 不在 GT 区域内
        gt_set = set(gt_token_indices)
        non_gt_text_indices = [idx for idx in text_token_indices if idx not in gt_set]

        print(f"[GT Mask] GT 区域: {len(gt_token_indices)} patches, "
              f"非GT文字区域: {len(non_gt_text_indices)} patches, "
              f"背景: {len(irrelevant_img_indices)} patches")
    else:
        # 无 GT 标注时，非GT文字区域 = 全部文字区域（保持向后兼容）
        non_gt_text_indices = list(text_token_indices)
        print("[GT Mask] 未提供 GT 标注，跳过 GT 区域分析。")

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

    # 绘制trend图 (传入两次注意力结果 + GT区域标注)
    plot_attention_trend_across_layers(
        attentions,
        attentions_prime,
        query_idx_q,
        query_idx_prime,
        image_indices,
        text_token_indices,
        irrelevant_img_indices,
        gt_token_indices,
        non_gt_text_indices
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

    # ---- GT 区域可视化：绿色 = GT bbox 关键区域 ----
    if bbox_json_path is not None and os.path.exists(bbox_json_path):
        gt_mask_resized = cv2.resize(gt_mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
        gt_overlay_img = image_bgr.copy()
        gt_overlay_img[mask_resized == 1] = [0, 0, 255]      # 红色 = 背景
        gt_overlay_img[gt_mask_resized == 1] = [0, 255, 0]   # 绿色 = GT 关键区域
        gt_overlay_img = cv2.addWeighted(gt_overlay_img, 0.6, image_bgr, 0.4, 0)
        gt_save_path = 'test_case/visualization/gt_region_overlay.jpg'
        cv2.imwrite(gt_save_path, gt_overlay_img)
        print(f"[GT Mask] GT区域可视化已保存至: {os.path.abspath(gt_save_path)}")

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
                                       text_token_indices, irrelevant_img_indices,
                                       gt_token_indices=None, non_gt_text_indices=None):
    """
    绘制 Relative Attention (A_q / A_q') 的层级变化趋势。
    新增 GT 区域 vs 非GT文字区域的分组对比，用于诊断模型是否聚焦在正确答案所需的区域上。
    """
    print("[Visualization] 正在计算并绘制每一层的 Relative Attention 变化趋势...")
    num_layers = len(attentions)
    epsilon = 1e-6

    # 映射回相对于图像块起点的局部索引 (0 到 num_img_tokens-1)
    image_start = image_indices[0]
    local_text_indices = [idx - image_start for idx in text_token_indices]
    local_bg_indices = [idx - image_start for idx in irrelevant_img_indices]

    has_gt = gt_token_indices is not None and len(gt_token_indices) > 0
    if has_gt:
        local_gt_indices = [idx - image_start for idx in gt_token_indices]
        local_non_gt_text_indices = [idx - image_start for idx in non_gt_text_indices]

    target_focus_scores = []
    bg_focus_scores = []
    gt_focus_scores = []
    non_gt_text_focus_scores = []

    for l in range(num_layers):
        # 第 l 层：真实问题的注意力 (q)
        layer_attn = attentions[l][0].mean(dim=0).cpu().float().numpy()
        q_to_img = layer_attn[query_idx_q][:, image_indices].mean(axis=0)

        # 第 l 层：通用问题的底噪注意力 (q')
        layer_attn_prime = attentions_prime[l][0].mean(dim=0).cpu().float().numpy()
        q_prime_to_img = layer_attn_prime[query_idx_prime][:, image_indices].mean(axis=0)

        # 核心：计算 Relative Attention
        rel_attn = q_to_img / (q_prime_to_img + epsilon)

        # HSV 文字区域和背景
        target_score = rel_attn[local_text_indices].mean()
        bg_score = rel_attn[local_bg_indices].mean()
        target_focus_scores.append(target_score)
        bg_focus_scores.append(bg_score)

        # GT 区域 vs 非GT文字区域
        if has_gt:
            gt_score = rel_attn[local_gt_indices].mean()
            non_gt_text_score = rel_attn[local_non_gt_text_indices].mean()
            gt_focus_scores.append(gt_score)
            non_gt_text_focus_scores.append(non_gt_text_score)

    # ================= 绘制趋势图 =================
    plt.figure(figsize=(14, 7))

    # 所有文字区域（红线）
    plt.plot(range(num_layers), target_focus_scores,
             label="All Text Regions (HSV)",
             color="#d62728", linewidth=2.5, marker='o', markersize=3)

    # 背景区域（蓝虚线）
    plt.plot(range(num_layers), bg_focus_scores,
             label="Background Regions",
             color="#1f77b4", linewidth=2, linestyle='--')

    if has_gt:
        # GT 真正关键区域（绿实线）
        plt.plot(range(num_layers), gt_focus_scores,
                 label="★ GT Relevant Regions (bbox annotated)",
                 color="#2ca02c", linewidth=3.0, marker='s', markersize=4,
                 markevery=max(1, num_layers // 10))

        # 非GT文字区域（橙虚线）—— 干扰文字
        plt.plot(range(num_layers), non_gt_text_focus_scores,
                 label="Non-GT Text Regions (irrelevant text)",
                 color="#ff7f0e", linewidth=2, linestyle='-.')

        # 打印关键层对比
        best_for_gt = np.argmax(gt_focus_scores[5:]) + 5
        print(f"[Diagnosis] GT 区域注意力最强的层: Layer {best_for_gt} "
              f"(Score: {gt_focus_scores[best_for_gt]:.4f})")
        print(f"[Diagnosis] 该层 GT vs 非GT文字 注意力差距: "
              f"{gt_focus_scores[best_for_gt] - non_gt_text_focus_scores[best_for_gt]:.4f}")
        if gt_focus_scores[best_for_gt] < non_gt_text_focus_scores[best_for_gt]:
            print("[Diagnosis] ⚠️ 模型聚焦在干扰文字上，而非GT关键区域！语义先验误导了视觉。")

    # 绘制 1.0 的基准线
    plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.4, label="Baseline (A_q = A_q')")

    # 标出 l* (HSV文字区域的视觉锚定最强层)
    best_layer = np.argmax(target_focus_scores[5:]) + 5
    plt.axvline(x=best_layer, color='gray', linestyle=':', alpha=0.7,
                label=f"Best Layer (l* = {best_layer})")

    plt.title("Relative Attention Trend (A_q / A_q') — Question vs Generic Attention",
              fontsize=14, fontweight='bold')
    plt.xlabel("Transformer Layer Depth", fontsize=12)
    plt.ylabel("Relative Attention Score", fontsize=12)
    plt.xticks(range(0, num_layers, 2))
    plt.legend(fontsize=9, loc='upper left')
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
    image_path = "test_case/test_images/image27.png"
    # image_path = "image2.jpg"
    image = Image.open(image_path).convert('RGB')

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # 这里只需放一个类型占位符
                {"type": "text",
                 "text": "2022年品牌一的收入占比是多少？"
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    # 3. 加载 GT bbox 标注路径
    bbox_json_path = "test_case/bbox_annotations/bbox_13.json"

    # 4. 前向传播：生成并提取 Full Attention（传入 GT 标注）
    with torch.no_grad():
        generated_ids = apply_zoomtext_and_glc(
            model, prompt, inputs, processor, image,
            bbox_json_path=bbox_json_path
        )
        # answer = processor.decode(
        #     generated_ids[0][inputs["input_ids"].shape[1]:],
        #     skip_special_tokens=True
        # )
