"""
分析 VLM (Gemma 4 12B) 在推理过程中对图像 token 和文本 token 的注意力行为。

核心论点:
  模型在复杂图像推理中频繁调用图像信息（持续性），但调用方式是
  全局弥散的（高熵、低集中度），而非显式定位到特定 patch（低熵、高集中度）。

分析 1：逐层 per-token 平均注意力对比
  图像 vs 文本的 per-key-token 平均注意力，x = 层深度。
  → 证明模型在所有层都持续消费图像信息。

分析 2：逐输出 token 的图像注意力行为（层间平均）
  三子图：(上) per-token 平均注意力  (中) 熵  (下) Top-5 集中度
  → 证明注意力持续存在但始终弥散，没有精确定位行为。

分析 3：最佳层的图像注意力分布直方图
  → 直观展示注意力值集中在均值附近，无长尾热点。

Gemma 4 12B 适配说明:
  - 模型: Gemma4UnifiedForConditionalGeneration (48 层标准 transformer)
  - 40 层 sliding window attention (local, window=1024) + 8 层 full attention (global)
  - 图像 token: <|image|> (id=258880) × N, 包围在 id=255999/258882 之间
  - 对话格式: <bos><|turn>user\n<|image|>...<turn|>\n<|turn>model\n...
  - Encoder-free: 图像通过 35M 线性投影直接进入 LLM embedding
  - 图像 token 使用双向注意力 (use_bidirectional_attention='vision')!
  - 使用 4-bit 量化以减少显存占用
"""

import torch
import os
import io
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import AutoProcessor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Gemma 4 特殊 token ID
# 图像: 起始标记 255999, patch 标记 258880, 结束标记 258882
# 高质量图像范围 (id >= 255000)
IMAGE_TOKEN_IDS = {255999, 258880, 258882}
IMAGE_PATCH_ID = 258880


# ============================================================
# Step 1: Token 分组识别 (Gemma 4 版本)
# ============================================================
def identify_token_groups(input_ids, processor):
    """
    对于一次完整的前向传播（question + answer 拼接），识别三类 token:
      image_indices:      图像 patch token 的位置
      text_indices_all:   所有非图像 token（prompt 文本 + chat template + 输出答案）
      output_indices:     assistant/model 部分（生成输出的 token）

    Gemma 4 特殊 token 结构:
      <bos> <|turn>user \n <|image|> × N <image|> question <turn|> \n <|turn>model \n ...
           token ids:       255999   258880×N  258882  text      106     105,4368
    """
    input_ids_list = input_ids if isinstance(input_ids, list) else list(input_ids)

    # 找图像边界：第一个 255999 到最后一个 258880（跳过末尾的 258882）
    image_start, image_end = -1, -1
    for i, tid in enumerate(input_ids_list):
        if tid == 255999:
            image_start = i
        elif tid == 258882 and image_start != -1:
            image_end = i
            break

    if image_start == -1 or image_end == -1:
        raise ValueError(f"Cannot locate image token boundaries: start={image_start}, end={image_end}")

    # 图像 token 包括起始标记、所有 patch 和结束标记
    image_indices = list(range(image_start, image_end + 1))

    all_set = set(range(len(input_ids_list)))
    text_indices_all = sorted(list(all_set - set(image_indices)))

    # 找 model/assistant turn 的位置: <|turn> (105) 后紧跟 model (4368)
    assistant_idx = -1
    for i, tid in enumerate(input_ids_list):
        if tid == 105 and i + 1 < len(input_ids_list):
            next_tok = processor.decode([input_ids_list[i + 1]])
            if 'model' in next_tok.lower():
                assistant_idx = i
                break

    if assistant_idx == -1:
        raise ValueError("Cannot locate <|turn>model token")

    output_indices = list(range(assistant_idx, len(input_ids_list)))

    return image_indices, text_indices_all, output_indices, image_start, image_end


# ============================================================
# Step 2: 注意力分析
# ============================================================
def analyze_attention(attentions, image_indices, text_indices_all, output_indices):
    """
    对每个 output token，计算三类指标（均为层间平均）:
    (a) per-key-token 平均注意力 (image 组 vs text 组)
    (b) 图像注意力的熵 (弥散程度)
    (c) 图像注意力的 Top-5 集中度 (聚焦程度)

    Gemma 4: attentions[l] 形状为 [batch, heads, seq_len, seq_len]
    48 层全部返回，包括 sliding window 和 full attention
    """
    num_layers = len(attentions)
    n_output = len(output_indices)

    layer_img_mean = np.zeros(num_layers, dtype=np.float32)
    layer_txt_mean = np.zeros(num_layers, dtype=np.float32)

    token_img_sum = np.zeros(n_output, dtype=np.float64)
    token_txt_sum = np.zeros(n_output, dtype=np.float64)
    token_entropy_sum = np.zeros(n_output, dtype=np.float64)
    token_top5_sum = np.zeros(n_output, dtype=np.float64)

    for l in range(num_layers):
        attn_l = attentions[l][0].mean(dim=0).cpu().float().numpy()  # [seq_len, seq_len]

        # ---- per-key-token 平均注意力 ----
        mean_to_image = attn_l[output_indices][:, image_indices].mean(axis=1)
        mean_to_text  = attn_l[output_indices][:, text_indices_all].mean(axis=1)

        layer_img_mean[l] = float(mean_to_image.mean())
        layer_txt_mean[l] = float(mean_to_text.mean())

        token_img_sum += mean_to_image
        token_txt_sum += mean_to_text

        # ---- 图像注意力分布：熵 & Top-5 集中度 ----
        img_attn = attn_l[output_indices][:, image_indices]  # [n_output, n_image]
        row_sums = img_attn.sum(axis=1, keepdims=True)
        img_dist = img_attn / (row_sums + 1e-12)

        entropy = -(img_dist * np.log(img_dist + 1e-12)).sum(axis=1)
        token_entropy_sum += entropy

        sorted_dist = np.sort(img_dist, axis=1)[:, ::-1]
        top5 = sorted_dist[:, :5].sum(axis=1)
        token_top5_sum += top5

    token_img_mean  = token_img_sum  / num_layers
    token_txt_mean  = token_txt_sum  / num_layers
    token_entropy   = token_entropy_sum / num_layers
    token_top5      = token_top5_sum    / num_layers

    max_entropy = np.log(len(image_indices))

    return (layer_img_mean, layer_txt_mean,
            token_img_mean, token_txt_mean,
            token_entropy, token_top5, max_entropy)


# ============================================================
# Step 3-7: 可视化函数（与原版一致，略作标注调整）
# ============================================================
def plot_analysis_1_layerwise(layer_img_mean, layer_txt_mean, save_path, layer_types=None):
    """逐层：每个 image/text token 的平均注意力对比"""
    num_layers = len(layer_img_mean)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    x = range(num_layers)
    ax.plot(x, layer_img_mean,
            label="mean attn per image key",
            color="#d62728", linewidth=2.5, marker='o', markersize=3)
    ax.plot(x, layer_txt_mean,
            label="mean attn per text key",
            color="#1f77b4", linewidth=2.5, marker='s', markersize=3)

    # 标注 full attention 层位置
    if layer_types:
        full_layers = [i for i, lt in enumerate(layer_types) if lt == 'full_attention']
        for fl in full_layers:
            ax.axvline(x=fl, color='green', linestyle='--', alpha=0.2, linewidth=1)

    best_img = np.argmax(layer_img_mean)
    best_txt = np.argmax(layer_txt_mean)
    ax.axvline(x=best_img, color="#d62728", linestyle=':', alpha=0.4)
    ax.axvline(x=best_txt, color="#1f77b4", linestyle=':', alpha=0.4)

    ax.set_xlabel("Transformer Layer (Gemma 4, 48 layers: 40 sliding + 8 full)", fontsize=12)
    ax.set_ylabel("Mean Attention per Key Token", fontsize=12)
    ax.set_title("Average Attention along Layers — Image vs Text\n(Gemma 4 12B, all 48 layers)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[分析1] 已保存至: {os.path.abspath(save_path)}")


def plot_analysis_2_tokenwise(token_img_mean, token_txt_mean,
                               token_entropy, max_entropy,
                               save_path):
    """双子图：per-token 平均注意力 + 熵"""
    n_tokens = len(token_img_mean)
    x = range(n_tokens)
    n_image_est = int(np.round(np.exp(max_entropy)))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    (ax1, ax2) = axes

    ax1.plot(x, token_img_mean, label="mean attn per image token",
             color="#d62728", linewidth=1.8)
    ax1.plot(x, token_txt_mean, label="mean attn per text token",
             color="#1f77b4", linewidth=1.8)
    ax1.set_ylabel("Mean Attention per Key Token", fontsize=11)
    ax1.set_title("(A) Average Attention along Tokens — Image vs Text",
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, token_entropy, color="#9467bd", linewidth=1.2, alpha=0.9)
    ax2.axhline(y=max_entropy, color='gray', linestyle='--', alpha=0.5, linewidth=1.2,
                label=f"max = ln({n_image_est}) = {max_entropy:.2f} (perfectly uniform)")
    half_entropy = max_entropy * 0.5
    ax2.axhline(y=half_entropy, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    e_min, e_max = token_entropy.min(), token_entropy.max()
    ax2.annotate(f'range: [{e_min:.1f}, {e_max:.1f}]\n'
                 f'= [{e_min/max_entropy:.0%}, {e_max/max_entropy:.0%}] of max\n'
                 f'always near max → diffuse',
                xy=(n_tokens * 0.98, e_max), fontsize=9, color='#9467bd',
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.set_xlabel("Output Token Position", fontsize=12)
    ax2.set_ylabel("Entropy (nats)", fontsize=11)
    ax2.set_title("(B) Image Attention Entropy", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[分析2] 已保存至: {os.path.abspath(save_path)}")


def plot_analysis_3_distribution(attentions, output_indices, image_indices,
                                  best_img_layer, save_path):
    """分布直方图 + CDF"""
    num_layers = len(attentions)

    img_attn_all = np.zeros((len(output_indices), len(image_indices)), dtype=np.float64)
    for l in range(num_layers):
        attn_l = attentions[l][0].mean(dim=0).cpu().float().numpy()
        img_attn_all += attn_l[output_indices][:, image_indices]
    img_attn_all /= num_layers

    row_sums = img_attn_all.sum(axis=1, keepdims=True)
    img_dist = img_attn_all / (row_sums + 1e-12)

    all_vals = img_dist.flatten()
    all_vals = all_vals[all_vals > 1e-8]

    n_image = len(image_indices)
    n_output = len(output_indices)
    uniform_val = 1.0 / n_image

    cv = all_vals.std() / (all_vals.mean() + 1e-12)
    below = float((all_vals <= uniform_val).mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.hist(all_vals, bins=150, color="#2ca02c", alpha=0.6,
             edgecolor='white', linewidth=0.2, log=True)
    ax1.axvline(x=uniform_val, color='red', linestyle='--', linewidth=2.5,
                label=f'uniform = 1/{n_image} = {uniform_val:.4f}')
    ax1.set_xlabel("Normalized attention per (token, patch)", fontsize=11)
    ax1.set_ylabel("Count (log scale)", fontsize=11)
    ax1.set_title(f"Distribution\n{n_output} tokens x {n_image} patches | CV={cv:.2f} | {below:.0%} below uniform",
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)

    sorted_vals = np.sort(all_vals)
    cdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    p90 = np.percentile(all_vals, 90)
    p99 = np.percentile(all_vals, 99)

    ax2.plot(sorted_vals, cdf_y, color="#2ca02c", linewidth=2.5)
    ax2.axvline(x=uniform_val, color='red', linestyle='--', linewidth=2.5,
                label=f'uniform = {uniform_val:.4f}')
    ax2.axvline(x=p90, color='#ff7f0e', linestyle=':', linewidth=2,
                label=f'P90 = {p90:.4f}')
    ax2.axvline(x=p99, color='#d62728', linestyle=':', linewidth=2,
                label=f'P99 = {p99:.4f}  ({p99/uniform_val:.1f}x)')
    ax2.set_xlabel("Normalized attention per (token, patch)", fontsize=11)
    ax2.set_ylabel("Cumulative probability", fontsize=11)
    ax2.set_title("CDF", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[分析3] 已保存至: {os.path.abspath(save_path)}")
    print(f"[分析3] uniform={uniform_val:.6f}  CV={cv:.2f}  below_uniform={below:.0%}")


def plot_analysis_4_token_attention_heatmap(attentions, best_img_layer, input_ids,
                                              image_indices, output_indices,
                                              image_start, image_end, assistant_idx,
                                              processor, save_path):
    """Token 间注意力热力图"""
    import matplotlib.colors as mcolors

    num_layers = len(attentions)
    seq_len_0 = attentions[0][0].shape[2]
    attn = np.zeros((seq_len_0, seq_len_0), dtype=np.float64)
    for l in range(num_layers):
        attn += attentions[l][0].mean(dim=0).cpu().float().numpy()
    attn /= num_layers

    seq_len = attn.shape[0]
    n_img = len(image_indices)
    n_out = len(output_indices)

    prompt_text_start = image_end + 1
    prompt_text_end = assistant_idx
    output_start = assistant_idx
    output_end = seq_len

    boundaries = [0, image_start, image_end + 1, prompt_text_end, output_end]
    boundary_labels = [
        "sys/pre",
        f"image\n({n_img} patches)",
        f"prompt text\n({prompt_text_end - prompt_text_start} tokens)",
        f"generated output\n({n_out} tokens)",
    ]
    boundary_colors = ['#aaaaaa', '#2ca02c', '#1f77b4', '#d62728']

    # 完整矩阵
    fig, axes = plt.subplots(1, 2, figsize=(22, 10),
                              gridspec_kw={'width_ratios': [20, 1]})
    ax, cax = axes

    attn_clipped = np.clip(attn, 1e-9, None)
    norm = mcolors.LogNorm(vmin=max(attn_clipped.min(), 1e-8), vmax=attn_clipped.max())

    im = ax.imshow(attn_clipped, cmap='inferno', norm=norm, aspect='auto',
                   origin='lower', interpolation='none')
    for b in boundaries:
        ax.axhline(y=b, color='white', linewidth=0.8, alpha=0.6, linestyle='-')
        ax.axvline(x=b, color='white', linewidth=0.8, alpha=0.6, linestyle='-')
    ax.plot([0, seq_len], [0, seq_len], color='white', linewidth=0.4, alpha=0.3, linestyle=':')

    ax.set_xlabel("Key Token Position", fontsize=13)
    ax.set_ylabel("Query Token Position", fontsize=13)
    ax.set_title(f"Token-to-Token Attention Matrix ({num_layers} layers avg, log scale)\n"
                 f"query (y) -> key (x), seq_len={seq_len}, image={n_img}, output={n_out}",
                 fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Attention Strength (log scale)", fontsize=12)

    for i, (b0, b1, label, color) in enumerate(zip(boundaries[:-1], boundaries[1:], boundary_labels, boundary_colors)):
        y_mid = (b0 + b1) / 2
        ax.text(-seq_len * 0.06, y_mid, label, ha='right', va='center', fontsize=8,
                color=color, fontweight='bold', transform=ax.transData,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.15,
                          edgecolor=color, linewidth=0.8))

    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, seq_len)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[分析4a] 已保存至: {os.path.abspath(save_path)}")

    # output-only 矩阵
    fig2, axes2 = plt.subplots(1, 2, figsize=(22, 8),
                                gridspec_kw={'width_ratios': [20, 1]})
    ax2, cax2 = axes2
    attn_output = attn_clipped[output_start:, :]

    im2 = ax2.imshow(attn_output, cmap='inferno', norm=norm, aspect='auto',
                     origin='lower', interpolation='none')
    for b in boundaries:
        ax2.axvline(x=b, color='white', linewidth=0.8, alpha=0.6, linestyle='-')

    for i, (b0, b1, label, color) in enumerate(zip(boundaries[:-1], boundaries[1:], boundary_labels, boundary_colors)):
        x_mid = (b0 + b1) / 2
        ax2.text(x_mid, n_out * 1.05, label.replace('\n', ' '), ha='center', va='bottom',
                fontsize=8, color=color, fontweight='bold', rotation=45)

    for pos, label in [(0, "start"), (n_out // 2, "middle"), (n_out - 1, "end")]:
        ax2.text(-seq_len * 0.04, pos, f"generation\n{label}",
                ha='right', va='center', fontsize=8, color='#d62728', fontweight='bold')

    ax2.set_xlabel("Key Token Position", fontsize=13)
    ax2.set_ylabel("Generated Token Position (query)", fontsize=13)
    ax2.set_title(f"Generated Tokens -> All Keys ({num_layers} layers avg, log scale)\n"
                  f"{n_out} generated tokens attending to {seq_len} key tokens",
                  fontsize=13, fontweight='bold')

    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label("Attention Strength (log scale)", fontsize=12)
    ax2.set_xlim(0, seq_len)
    ax2.set_ylim(0, n_out)
    plt.tight_layout()

    fig2_path = save_path.replace('.png', '_output_only.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[分析4b] 已保存至: {os.path.abspath(fig2_path)}")

    # 控制台：区块平均
    n_blocks = len(boundaries) - 1
    block_matrix = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            q_start, q_end = boundaries[i], boundaries[i + 1]
            k_start, k_end = boundaries[j], boundaries[j + 1]
            block = attn[q_start:q_end, k_start:k_end]
            block_matrix[i, j] = block.mean()

    print("\n" + "=" * 70)
    print(f"BLOCK-AVERAGED ATTENTION (all {num_layers} layers avg)")
    print("rows=query group,  cols=key group,  value=mean attention")
    print("=" * 70)
    header = f"{'query / key':<16}" + "".join(f"{lbl:>14}" for lbl in ['sys/pre', 'image', 'prompt', 'output'])
    print(header)
    print("-" * 70)
    for i in range(n_blocks):
        row = f"{boundary_labels[i].replace(chr(10),' '):<16}"
        for j in range(n_blocks):
            row += f"{block_matrix[i, j]:>14.6f}"
        print(row)
    print("-" * 70)
    out_to_img = block_matrix[3, 1] if n_blocks > 3 else 0
    out_to_prompt = block_matrix[3, 2] if n_blocks > 3 else 0
    out_to_out = block_matrix[3, 3] if n_blocks > 3 else 0
    print(f"  Output->Image cross-attn:  {out_to_img:.6f}")
    print(f"  Output->Prompt cross-attn: {out_to_prompt:.6f}")
    print(f"  Output->Output self-attn:  {out_to_out:.6f}")
    print(f"  Output: image/prompt ratio = {out_to_img/(out_to_prompt+1e-8):.4f}")
    print(f"  Output: output/image ratio = {out_to_out/(out_to_img+1e-8):.1f}x")


def plot_analysis_5_patch_heatmap(attentions, output_indices, image_indices,
                                    image, inputs, save_path):
    """
    Patch 注意力热力图叠加原图。

    Gemma 4: encoder-free 架构，图像 token 直接投影到 LLM embedding。
    使用 mm_token_type_ids 来确定图像 token 的 2D 空间排列。
    """
    num_layers = len(attentions)
    n_img = len(image_indices)
    n_out = len(output_indices)

    # 所有层平均的 output→image 交叉注意力
    img_attn_all = np.zeros((n_out, n_img), dtype=np.float64)
    for l in range(num_layers):
        attn_l = attentions[l][0].mean(dim=0).cpu().float().numpy()
        img_attn_all += attn_l[output_indices][:, image_indices]
    img_attn_all /= num_layers

    # Gemma 4 encoder-free: 使用 pixel_values 的 patch 数量确定 2D grid
    # pixel_values: [batch, n_pv_patches, features]
    # image tokens: 1 start + N patches + 1 end  (其中 N 可能与 n_pv_patches 不同)
    import math
    pv_shape = inputs['pixel_values'].shape  # e.g. [1, 280, 6912]
    n_pv_patches = pv_shape[1]

    # 跳过首尾标记，只取中间的 patch token
    patch_start = 1       # 跳过 id=255999
    patch_end = n_img - 1  # 跳过末尾 id=258882
    n_patch_tokens = patch_end - patch_start

    # 用 pixel_values 的 patch 数推算 grid（encoder-free 架构以此为准）
    # 根据图像 aspect ratio 找最佳 grid
    img_w, img_h = image.size
    best_h, best_w = 1, n_pv_patches
    best_err = float('inf')
    for h in range(1, int(math.sqrt(n_pv_patches)) + 2):
        if n_pv_patches % h == 0:
            w = n_pv_patches // h
            err = abs(w/h - img_w/img_h)
            if err < best_err:
                best_err = err
                best_h, best_w = h, w
    vis_h, vis_w = best_h, best_w
    grid_cells = vis_h * vis_w

    print(f"[分析5] pixel_values patches={n_pv_patches}, image tokens={n_patch_tokens}")
    print(f"[分析5] 推断 grid: {vis_h}x{vis_w} (aspect match err={best_err:.3f})")

    # 如果 token 数少于 grid 格数，用零填充
    if n_patch_tokens < grid_cells:
        print(f"[分析5] token 数 ({n_patch_tokens}) < grid ({grid_cells}), 将用零填充空位")

    patch_attn_prompt = img_attn_all[:, patch_start:patch_end].mean(axis=0)

    # 动态划分
    split_point = n_out // 2
    patch_attn_early = img_attn_all[:split_point, patch_start:patch_end].mean(axis=0)
    patch_attn_late  = img_attn_all[split_point:, patch_start:patch_end].mean(axis=0)

    # 原图准备
    image_np = np.array(image.convert('RGB'), dtype=np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    def make_overlay(patch_attn, vmax_val=None):
        # 如果 token 少于 grid 格数，填充零到完整 grid
        if len(patch_attn) < grid_cells:
            padded = np.zeros(grid_cells, dtype=patch_attn.dtype)
            padded[:len(patch_attn)] = patch_attn
            attn_2d = padded.reshape(vis_h, vis_w)
        else:
            attn_2d = patch_attn[:grid_cells].reshape(vis_h, vis_w)

        fig, ax = plt.subplots(figsize=(max(vis_w / 4, 6), max(vis_h / 4, 3)))
        plt.axis('off')
        sns.heatmap(attn_2d, cmap='jet', cbar=False, vmax=vmax_val,
                    xticklabels=False, yticklabels=False, square=True, ax=ax)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        heatmap_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        heatmap_bgr = cv2.imdecode(heatmap_array, cv2.IMREAD_COLOR)
        if heatmap_bgr is None:
            raise ValueError("Failed to decode heatmap from memory!")
        heatmap_resize = cv2.resize(heatmap_bgr, (w, h))
        combined_bgr = cv2.addWeighted(heatmap_resize, 0.5, image_bgr, 0.7, 0)
        return cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

    p95_early = np.percentile(patch_attn_early, 95)
    p95_late = np.percentile(patch_attn_late, 95)
    common_vmax = max(p95_early, p95_late)

    overlay_early = make_overlay(patch_attn_early, vmax_val=common_vmax)
    overlay_late  = make_overlay(patch_attn_late,  vmax_val=common_vmax)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    e_mean = patch_attn_early.mean()
    l_mean = patch_attn_late.mean()

    axes[0].imshow(overlay_early)
    axes[0].set_title(f"Early tokens (0-{split_point - 1})\nper-patch mean = {e_mean:.6f}",
                      fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(overlay_late)
    axes[1].set_title(f"Later tokens ({split_point}-{n_out - 1})\nper-patch mean = {l_mean:.6f}",
                      fontsize=11, fontweight='bold')
    axes[1].axis('off')

    fig.text(0.5, -0.02,
             f"Image: {w}x{h}  |  Grid: {vis_h}x{vis_w} ({n_patch_tokens} patches)"
             f"  |  {num_layers} layers avg"
             f"  |  vmax={common_vmax:.6f} (P95)"
             f"  |  seaborn heatmap + cv2.addWeighted overlay",
             ha='center', fontsize=9, color='gray', style='italic')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[分析5] 已保存至: {os.path.abspath(save_path)}")
    print(f"[分析5] Grid: {vis_h}x{vis_w} ({n_patch_tokens} patches) | "
          f"early mean={e_mean:.6f} | late mean={l_mean:.6f} | vmax(P95)={common_vmax:.6f}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    # 1. 加载模型 (Gemma 4 12B, 4-bit)
    model_path = os.path.join(SCRIPT_DIR, "local_google_gemma-4-12b-it")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"Loading Gemma 4 12B model from {model_path}...")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # processor IS the tokenizer for Gemma
    tokenizer = getattr(processor, 'tokenizer', processor)

    from transformers import AutoModelForImageTextToText

    # Gemma 4 12B 的视觉模块 LayerNorm 与 4-bit 量化不兼容
    # 使用 bf16 全精度加载，device_map="auto" 自动 offload 到 CPU
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded: {type(model).__name__}")

    # 2. 加载图片和问题
    image_path = os.path.join(SCRIPT_DIR, "test_case/test_images/image26.jpg")
    image = Image.open(image_path).convert('RGB')

    with open(os.path.join(SCRIPT_DIR, "test_case/bbox_annotations/bbox_9.json"),
              'r', encoding='utf-8') as f:
        bbox_data = json.load(f)
    question = bbox_data['question']
    gt_answer = bbox_data['ground_truth']

    # 3. 真实生成
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}
    ]
    gen_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    gen_inputs = processor(text=gen_prompt, images=[image], return_tensors="pt")
    try:
        gen_inputs = gen_inputs.to(model.device)
    except:
        gen_inputs = gen_inputs.to(device)

    prompt_len = gen_inputs["input_ids"].shape[1]
    print(f"Prompt len: {prompt_len} tokens")
    print("Generating...")

    eos_id = getattr(model.generation_config, 'eos_token_id', 1)  # <eos>
    pad_id = getattr(model.generation_config, 'pad_token_id', 0) or 0

    with torch.no_grad():
        gen_output = model.generate(
            **gen_inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=1.0,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    generated_ids = gen_output[0][prompt_len:].cpu().tolist()
    generated_text = processor.decode(generated_ids, skip_special_tokens=False)
    generated_text_clean = processor.decode(generated_ids, skip_special_tokens=True)
    n_generated = len(generated_ids)

    print(f"Generated {n_generated} tokens")
    print(f"Preview:\n{generated_text_clean[:300]}...\n")

    # 4. 完整序列前向传播
    messages_full = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": generated_text_clean}
        ]}
    ]
    full_prompt = processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=full_prompt, images=[image], return_tensors="pt")
    try:
        inputs = inputs.to(model.device)
    except:
        inputs = inputs.to(device)

    print(f"Full seq len: {inputs['input_ids'].shape[1]} tokens")
    print("Forward pass with output_attentions=True...")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    num_layers = len(attentions)
    input_ids = inputs["input_ids"][0].cpu().tolist()

    print(f"Attention layers: {num_layers} (all transformer layers)")

    # 5. 分组识别
    image_indices, text_indices_all, output_indices, img_start, img_end = \
        identify_token_groups(input_ids, processor)

    n_img = len(image_indices)
    n_out = len(output_indices)
    total_tokens = len(input_ids)

    print(f"Image tokens:   {n_img:>6}  (positions {img_start}-{img_end})")
    print(f"Text tokens:    {len(text_indices_all):>6}")
    print(f"Output tokens:  {n_out:>6}  (from <|turn>model)")
    print(f"Total layers:   {num_layers}")

    # 获取 layer_types 信息
    text_config = getattr(model.config, 'text_config', None)
    layer_types = getattr(text_config, 'layer_types', None)

    # 6. 分析
    (layer_img_mean, layer_txt_mean,
     token_img_mean, token_txt_mean,
     token_entropy, token_top5, max_entropy) = \
        analyze_attention(attentions, image_indices, text_indices_all, output_indices)

    # best image layer: 跳过前 3 层
    best_img_layer = int(np.argmax(layer_img_mean[3:]) + 3)

    # 7. 输出目录
    vis_dir = os.path.join(SCRIPT_DIR, "test_case/visualization_gemma4")
    os.makedirs(vis_dir, exist_ok=True)

    # 8. 绘图
    plot_analysis_1_layerwise(
        layer_img_mean, layer_txt_mean,
        os.path.join(vis_dir, "analysis1_layerwise_attn.png"),
        layer_types=layer_types
    )
    plot_analysis_2_tokenwise(
        token_img_mean, token_txt_mean,
        token_entropy, max_entropy,
        os.path.join(vis_dir, "analysis2_tokenwise_diffuse.png")
    )
    plot_analysis_3_distribution(
        attentions, output_indices, image_indices, best_img_layer,
        os.path.join(vis_dir, "analysis3_distribution.png")
    )
    plot_analysis_4_token_attention_heatmap(
        attentions, best_img_layer, input_ids,
        image_indices, output_indices,
        img_start, img_end, output_indices[0],
        processor,
        os.path.join(vis_dir, "analysis4_token_attention_heatmap.png")
    )
    plot_analysis_5_patch_heatmap(
        attentions, output_indices, image_indices,
        image, inputs,
        os.path.join(vis_dir, "analysis5_patch_attention_heatmap.png")
    )

    # 9. 数据输出
    avg_img = float(layer_img_mean.mean())
    avg_txt = float(layer_txt_mean.mean())
    avg_entropy = float(token_entropy.mean())
    avg_top5 = float(token_top5.mean())

    print("\n" + "=" * 70)
    print("SUMMARY (Gemma 4 12B)")
    print("=" * 70)
    print(f"Generated tokens: {n_generated}")
    print(f"Token counts — Image: {n_img}, Output: {n_out}, Total: {total_tokens}")
    print()

    # 9a. 逐层数据
    print("-" * 70)
    print(f"{'Layer':<8} {'ImgMean':>10} {'TxtMean':>10} {'Img/Txt':>10} {'Type':<18} {'Interpretation'}")
    print("-" * 70)
    for l in range(num_layers):
        ratio_l = layer_img_mean[l] / (layer_txt_mean[l] + 1e-8)
        if ratio_l > 1.0:
            interp = "prefers IMAGE"
        elif ratio_l > 0.8:
            interp = "≈ balanced"
        elif ratio_l > 0.3:
            interp = "leans text"
        else:
            interp = "strongly TEXT"
        marker = " ← BEST" if l == best_img_layer else ""
        lt = layer_types[l] if layer_types else "?"
        print(f"L{l:<7} {layer_img_mean[l]:>10.6f} {layer_txt_mean[l]:>10.6f} "
              f"{ratio_l:>10.3f}  {lt:<18} {interp}{marker}")
    print("-" * 70)
    print(f"{'avg':<8} {avg_img:>10.6f} {avg_txt:>10.6f} {avg_img/(avg_txt+1e-8):>10.3f}")
    print()

    # 9b. 持续性
    n = n_out
    seg_size = max(1, n // 3)
    seg1 = token_img_mean[:seg_size].mean()
    seg2 = token_img_mean[seg_size:2*seg_size].mean()
    seg3 = token_img_mean[2*seg_size:].mean()
    drop_ratio = (seg1 - seg3) / (seg1 + 1e-8)
    print(f"IMAGE ATTENTION PERSISTENCE: seg1={seg1:.6f} → seg3={seg3:.6f}")
    if drop_ratio > 0.3:
        print(f"  [WARN] Image attention drops {drop_ratio:.0%}")
    else:
        print(f"  [OK] Image attention stable")
    print()

    # 9c. 弥散性
    uniform_top5_exp = 5.0 / n_img
    print(f"IMAGE ATTENTION DIFFUSENESS:")
    print(f"  Entropy: mean={avg_entropy:.3f}, max=ln({n_img})={max_entropy:.3f}, "
          f"efficiency={avg_entropy/max_entropy:.1%}")
    print(f"  Top-5: mean={avg_top5*100:.2f}%, uniform expectation={uniform_top5_exp*100:.2f}%, "
          f"ratio={avg_top5/uniform_top5_exp:.1f}x")
    print()

    # 9d. 诊断
    img_txt_ratio = avg_img / (avg_txt + 1e-8)
    entropy_ratio = avg_entropy / max_entropy
    top5_ratio = avg_top5 / uniform_top5_exp

    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print(f"  1. Image attention MAGNITUDE: ratio={img_txt_ratio:.3f}")
    print(f"  2. MAGNITUDE TREND: seg1={seg1:.6f} → seg3={seg3:.6f}")
    print(f"  3. DISTRIBUTION: H={avg_entropy:.1f}/{max_entropy:.1f} ({entropy_ratio:.0%}), "
          f"Top-5={avg_top5*100:.0f}% ({top5_ratio:.0f}x uniform)")
    print(f"  4. Best visual anchoring layer: L{best_img_layer}")

    # 9e. JSON 保存
    results_json = {
        "model": "Gemma-4-12B-it",
        "architecture": "gemma4_unified, encoder-free, 48 layers (40 sliding + 8 full)",
        "config": {
            "image": image_path,
            "question": question,
            "generated_text": generated_text_clean,
            "n_image_tokens": n_img,
            "n_output_tokens": n_out,
            "total_tokens": total_tokens,
            "n_layers": num_layers,
            "best_img_layer": best_img_layer,
            "max_entropy": float(max_entropy),
        },
        "layer_wise": {
            "image_mean": layer_img_mean.tolist(),
            "text_mean": layer_txt_mean.tolist(),
            "image_text_ratio": (layer_img_mean / (layer_txt_mean + 1e-8)).tolist(),
            "layer_types": layer_types,
        },
        "token_wise": {
            "image_mean": token_img_mean.tolist(),
            "text_mean": token_txt_mean.tolist(),
            "entropy": token_entropy.tolist(),
            "top5_concentration": token_top5.tolist(),
        },
        "summary": {
            "avg_image_mean": avg_img,
            "avg_text_mean": avg_txt,
            "image_text_ratio": float(img_txt_ratio),
            "avg_entropy": avg_entropy,
            "entropy_ratio_of_max": float(entropy_ratio),
            "avg_top5_pct": float(avg_top5 * 100),
            "top5_ratio_to_uniform": float(top5_ratio),
            "seg1_image_mean": float(seg1),
            "seg2_image_mean": float(seg2),
            "seg3_image_mean": float(seg3),
        }
    }
    json_path = os.path.join(vis_dir, "attention_analysis_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    print(f"\n[Data] Results saved to: {os.path.abspath(json_path)}")
    print("Done!")
