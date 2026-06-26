import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, Glm4vForConditionalGeneration


def get_color_based_background_mask(image_pil, vis_h, vis_w):
    """
    通过 HSV 色彩空间，物理提取图表的纯背景 Patch。
    返回一个 (vis_h, vis_w) 的二维二值掩码 (1=背景, 0=图文)
    """
    image_np = np.array(image_pil.convert('RGB'), dtype=np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    bg_pixel_mask = ((s < 35) & (v > 200)).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    fg_pixel_mask = 1 - bg_pixel_mask
    fg_pixel_mask = cv2.dilate(fg_pixel_mask, kernel, iterations=1)
    bg_pixel_mask = 1 - fg_pixel_mask

    mask_2d = np.zeros((vis_h, vis_w), dtype=np.uint8)
    patch_h = h / vis_h
    patch_w = w / vis_w

    for i in range(vis_h):
        for j in range(vis_w):
            y1, y2 = int(i * patch_h), int((i + 1) * patch_h)
            x1, x2 = int(j * patch_w), int((j + 1) * patch_w)

            patch_pixels = bg_pixel_mask[y1:y2, x1:x2]
            bg_ratio = np.sum(patch_pixels) / (patch_pixels.size + 1e-6)

            if bg_ratio > 0.90:
                mask_2d[i, j] = 1
            else:
                mask_2d[i, j] = 0

    return mask_2d


def extract_attention_for_prompt(model, processor, image, text_prompt):
    """
    运行一次完整的前向传播，提取给定 prompt 的 Attention 矩阵
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        # 阶段一：自回归生成答案
        generate_kwargs = {"max_new_tokens": 512, "do_sample": False, "temperature": 0.0}
        generated_ids = model.generate(**inputs, **generate_kwargs)

        # 阶段二：回放全量序列，提取 Attention
        full_input_ids = generated_ids[0].unsqueeze(0)
        forward_kwargs = {
            "input_ids": full_input_ids,
            "attention_mask": torch.ones_like(full_input_ids),
            "output_attentions": True,
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw
        }

        # 将 mm_token_type_ids 提取出来，并为新生成的 Answer Tokens 补 0 (代表纯文本)
        if "mm_token_type_ids" in inputs:
            original_mm_type = inputs["mm_token_type_ids"]
            added_seq_len = full_input_ids.shape[1] - original_mm_type.shape[1]

            if added_seq_len > 0:
                # 用 0 填充新生成的文本部分
                padding_zeros = torch.zeros(
                    (1, added_seq_len),
                    dtype=original_mm_type.dtype,
                    device=original_mm_type.device
                )
                padded_mm_type = torch.cat([original_mm_type, padding_zeros], dim=1)
                forward_kwargs["mm_token_type_ids"] = padded_mm_type
            else:
                forward_kwargs["mm_token_type_ids"] = original_mm_type
        outputs = model(**forward_kwargs)

    attentions = outputs.attentions
    tokens = [processor.decode([tok]) for tok in full_input_ids[0]]

    # 动态寻找 GLM 图像 Token 边界
    image_start, image_end = -1, -1
    for i, token in enumerate(tokens):
        if token == '<|begin_of_image|>':
            image_start = i + 1
        elif token == '<|end_of_image|>':
            image_end = i
            break

    if image_start == -1 or image_end == -1:
        image_start, image_end = 5, 1205


    # 获取原始 Prompt 的长度（不包含模型后来生成的答案）
    prompt_len = inputs["input_ids"].shape[1]

    starting_answer_token_idx = prompt_len - 1

    # 将 Query 索引严格收束为这 1 个 Token
    query_indices = [starting_answer_token_idx]

    image_indices = list(range(image_start, image_end))

    return attentions, query_indices, image_indices


def run_relative_attention_analysis():
    # 1. 加载本地模型
    model_path = "local_GLM-4.6V-Flash"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_path} with EAGER attention...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    ).eval()

    # 2. 读取图片并计算 HSV 区域掩码 (确保纯粹的 Token 分类)
    image_path = "test_case/test_images/image0.jpg"
    image = Image.open(image_path).convert('RGB')

    dummy_inputs = processor(images=image, text="", return_tensors="pt")
    grid_thw = dummy_inputs["image_grid_thw"][0]
    vis_h, vis_w = grid_thw[1].item() // 2, grid_thw[2].item() // 2

    print(f"\n[HSV Mask] 正在通过色彩空间提取背景... (网格: {vis_h}x{vis_w})")
    mask_2d = get_color_based_background_mask(image, vis_h, vis_w)
    mask_1d = mask_2d.flatten()

    # 映射为 Image 内部相对索引（0 到 num_img_tokens-1）
    text_local_indices = np.where(mask_1d == 0)[0].tolist()
    bg_local_indices = np.where(mask_1d == 1)[0].tolist()
    print(f"[HSV Mask] 锁定 {len(text_local_indices)} 个文字/图表 Patch，{len(bg_local_indices)} 个背景 Patch。")

    # 3. 提取实际问题 (q) 的 Attention
    # q = "Please pay close attention to the textual information in the image, as well as the key elements specified in the question, such as the objects, relationships, and constraints. Then answer the following question within the tags <q> and </q> step-by-step according to the textual content, explaining your reasoning process clearly at each stage and the text clues you use. After presenting your reasoning, provide the final simple answer at last, making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. Question: <q>What is the average number of the genres of movies made by directors who have worked in drama genres?</q> \n Answer"
    q = "What is the average number of the genres of movies made by directors who have worked in drama genres? Answer"
    print(f"\n[Attention] 正在推理真实问题 (q) 并提取注意力矩阵...")
    attn_q, query_idx_q, img_idx_q = extract_attention_for_prompt(model, processor, image, q)

    # 4. 提取论文建议的通用指令 (q') 的 Attention
    q_prime = "Write a general description of the image. Answer"
    print(f"[Attention] 正在推理通用描述 (q') 作为底噪矩阵...")
    attn_q_prime, query_idx_q_prime, img_idx_q_prime = extract_attention_for_prompt(model, processor, image, q_prime)

    # 5. 计算并归一化 Relative Attention
    num_layers = len(attn_q)
    target_focus_scores = []
    bg_focus_scores = []
    attention_ratios = []
    epsilon = 1e-6

    print("\n[Calculation] 正在执行逐层 Relative Attention (A_rel = A_q / A_q') 运算...")
    for l in range(num_layers):
        # 提取第 l 层真实问题的 Query-to-Image 关注度
        layer_q = attn_q[l][0].mean(dim=0).cpu().float().numpy()
        q_to_img = layer_q[query_idx_q][:, img_idx_q].mean(axis=0)  # Shape: (num_img_tokens,)

        # 提取第 l 层通用问题的 Query-to-Image 关注度 (底噪)
        layer_q_prime = attn_q_prime[l][0].mean(dim=0).cpu().float().numpy()
        q_prime_to_img = layer_q_prime[query_idx_q_prime][:, img_idx_q_prime].mean(axis=0)

        # 核心数学过滤：元素级除法计算相对注意力
        rel_attn = q_to_img / (q_prime_to_img + epsilon)

        # 分别计算文字区域和背景区域的平均相对分数
        target_score = rel_attn[text_local_indices].mean()
        bg_score = rel_attn[bg_local_indices].mean()
        global_score = rel_attn.mean()

        attn_ratio = target_score / (global_score + epsilon)

        target_focus_scores.append(target_score)
        bg_focus_scores.append(bg_score)
        attention_ratios.append(attn_ratio)

    # 6. 绘制去噪后的相对注意力趋势图
    plt.figure(figsize=(12, 6))

    plt.plot(range(num_layers), target_focus_scores,
             label="Relative Focus on Relevant Text Patches (A_rel > 1 means Signal)",
             color="#d62728", linewidth=2.5, marker='o', markersize=4)

    plt.plot(range(num_layers), bg_focus_scores,
             label="Relative Focus on Irrelevant Background",
             color="#1f77b4", linewidth=2, linestyle='--')

    plt.plot(range(num_layers), attention_ratios,
             label="Attention Ratio (Target Mean / Global Mean)",
             color="#2ca02c", linewidth=2.5, linestyle='-.', marker='x', markersize=4)

    # 标出峰值层 (避开前两层，寻找深层特征突变)
    search_start = 5
    best_layer = np.argmax(target_focus_scores[search_start:]) + search_start
    plt.axvline(x=best_layer, color='gray', linestyle=':', label=f"Best Layer (l* = {best_layer})")

    # 绘制 Y=1 的基准线 (1代表和闲聊没区别)
    plt.axhline(y=1.0, color='green', linestyle='-', alpha=0.3, label="Baseline (A_q = A_q')")

    plt.title("Relative Attention Trend (A_rel) Across Layers [Based on ICLR 2025 Method]", fontsize=14,
              fontweight='bold')
    plt.xlabel("Transformer Layer Depth", fontsize=12)
    plt.ylabel("Relative Attention Score (A_q / A_q')", fontsize=12)
    plt.xticks(range(0, num_layers, 2))
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    save_path = "test_case/visualization/relative_attention_trend_paper_method.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\n[Success] 相对注意力 (Relative Attention) 趋势图已保存至: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    run_relative_attention_analysis()