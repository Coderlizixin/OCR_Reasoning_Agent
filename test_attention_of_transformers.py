import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from PIL import Image
import cv2
import numpy as np
import io
from test_when_semantics_mislead_vision import apply_zoomtext_and_glc


# Question0: What is the average number of the genres of movies made by directors who have worked in drama genres?
# Question1: What is the proportion of French food among all foreign foods?
# Question2: Which country has the greatest Variance in terms of shopping opinions? Assume that there are 100 samples collected from each country.
# Question3: Do consumers aged 40-45 buy the most pens?
# Question4: 假如某人把海报中粉色框框住的商品都买两包，他要花多少钱？
# Question5: 红色字体的数字之和多少？
# Question6: How many times does the number '27.99' appear on the receipt?
# Question7: 回答图片中的问题。如果是选择题，请提供你认为正确的选项。如果是简答题，请提供你的答案。
# Question8: What's the sum of all yellow squares?


def response_of_VLM():
    model_path = "local_GLM-4.6V-Flash"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading GLM-4.6V-Flash with EAGER attention...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # 2. 构造 OCR 推理测试数据
    image_path = "image8.jpg"  # 准备一张包含金额或数字的测试图，比如发票
    image = Image.open(image_path).convert('RGB')

    image_path2 = "yellow.png"
    image2 = Image.open(image_path2).convert('RGB')

    # 多跳 promtp
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image"},
#                 {"type": "text",
#                  "text": '''You are a general-purpose Visual Reasoning Agent.
# To solve complex visual question within the tags <q> and </q>, you MUST use the ReAct format.
#
# Think step-by-step. For each step, output your thought process, the sub-question you need to answer, and then answer it before moving to the next.
#
# Format:
# Thought 1: [Analyze the current situation and what information is missing]
# Sub-Question 1: [A targeted question about a specific part of the image]
# Observation 1: [Look at the image and answer Sub-Question 1]
#
# Thought 2: [Based on Observation 1, what is the next step?]
# Sub-Question 2: [Next targeted question]
# Observation 2: [Look at the image again to answer]
#
# ... (Repeat as necessary) ...
#
# Thought N: [I have all the information needed.]
# Final Answer: [The ultimate answer to the user's prompt]
# Question: <q>What is the average number of the genres of movies made by directors who have worked in drama genres?</q>'''}
#             ]
#         }
#     ]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text",
                 "text": "Please pay close attention to the textual information in the image, as well as the key elements "
                         "specified in the question, such as the objects, relationships, and constraints. "
                         "Then answer the following question within the tags <q> and </q> step-by-step "
                         "according to the textual content, explaining your reasoning process clearly at each stage and the "
                         "text clues you use. Before tackling the problem, make a plan and break it down into two to "
                         "four progressive subproblems. The second image shows a yellow square; note the difference in color compared to the square in the first image. After presenting your reasoning, provide the final simple answer at last, "
                         "making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. "
                         "Question: <q>Among different services, what is the minimum variance observed in their pricing?</q>"}
            ]
        }
    ]

    # 第一步：只生成拼装好的纯文本 prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,  # 注意这里是 False，先不转 token
        add_generation_prompt=True
    )

    # 第二步：将 Prompt 和 PIL Image 对象一起交给 processor
    inputs = processor(
        text=prompt,
        images=[image, image2],  # 图片以列表形式传入
        # images=[image],
        return_tensors="pt"
    ).to(device)

    # 3. 前向传播：生成并提取 Attention
    print("Generating response and extracting attention...")
    with torch.no_grad():
        # 正常生成输出（包含了 <think> 过程和最终 <answer>）
        generate_kwargs = {"max_new_tokens": 8192, "do_sample": False, "temperature": 0.0}
        generated_ids = model.generate(**inputs, **generate_kwargs)
        # generated_ids = apply_zoomtext_and_glc(model, prompt, inputs, processor, image)

        answer = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        print("Answer:", answer)


def draw_text_attn():
    # 1. 基础配置与模型加载
    model_path = "local_GLM-4.1V_thinking"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading GLM-4.1V-Thinking with EAGER attention...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # 2. 构造 OCR 推理测试数据
    image_path = "test_case/test_images/image0.jpg"
    image = Image.open(image_path).convert('RGB')

    # 构建对话：强迫模型使用 CoT 进行多步计算
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
                         "text clues you use. Before tackling the problem, make a plan and break it down into two to "
                         "four progressive subproblems. After presenting your reasoning, provide the final simple answer at last, "
                         "making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. "
                         "Question: <q>What is the average number of the genres of movies made by directors who have worked in drama genres?</q>"}
            ]
        }
    ]

    # 第一步：只生成拼装好的纯文本 prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,  # 注意这里是 False，先不转 token
        add_generation_prompt=True
    )

    # 第二步：将 Prompt 和 PIL Image 对象一起交给 processor
    inputs = processor(
        text=prompt,
        images=[image],  # 图片以列表形式传入
        return_tensors="pt"
    ).to(device)

    # 3. 前向传播：生成并提取 Attention
    print("Generating response and extracting attention...")
    with torch.no_grad():
        # 正常生成输出（包含了 <think> 过程和最终 <answer>）
        generate_kwargs = {"max_new_tokens": 2048, "do_sample": False, "temperature": 0.0}
        # generated_ids = model.generate(**inputs, **generate_kwargs)
        generated_ids = apply_zoomtext_and_glc(model, prompt, inputs, processor, image)

        answer = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        print("Answer:", answer)

        # 将输入和生成的输出拼接成完整的序列
        full_input_ids = generated_ids[0].unsqueeze(0)

        # 构建包含完整序列的参数字典进行二次 Forward
        forward_kwargs = {
            "input_ids": full_input_ids,
            "attention_mask": torch.ones_like(full_input_ids),  # 必须提供匹配长度的 mask
            "output_attentions": True
        }

        # 必须把 processor 生成的所有视觉相关的参数都带上
        if "pixel_values" in inputs:
            forward_kwargs["pixel_values"] = inputs.pixel_values
        if "image_grid_thw" in inputs:
            forward_kwargs["image_grid_thw"] = inputs.image_grid_thw

        # 执行前向传播，提取特征
        outputs = model(**forward_kwargs)

    # 4. 获取注意力数据与 Token 映射
    attentions = outputs.attentions
    # 注意：GLM 的 tokenizer 解码器对特殊字符处理比较特殊，我们可以逐个解码观察
    tokens = [processor.decode([tok]) for tok in full_input_ids[0]]

    # 5. 定位并裁剪矩阵 (剔除图像 Token 区域)
    # 图像 Token 在 GLM 中通常被统一替换或占据前面一大块固定长度
    # 我们寻找人类可读的文本起始点，比如寻找提示词的起始特征
    text_start_idx = 0
    # for i, token in enumerate(tokens):
    #     if "读取图片" in token or "发票" in token or 'Please' in token:  # 根据你的 prompt 寻找锚点
    #         # 往前退一点，保留完整的文本 prompt
    #         text_start_idx = max(0, i - 5)
    #         break

    start_idx = text_start_idx
    clean_tokens = tokens[start_idx:]

    # 选择一个中深层进行观察（例如第 20 层，GLM-9B 通常有 40 层）
    layer_idx = 20
    # 取出该层，去掉 batch，沿 head 取平均
    layer_attn = attentions[layer_idx][0]
    avg_attn = layer_attn.mean(dim=0).cpu().float().numpy()

    # 裁剪 Attention 矩阵
    sliced_attn = avg_attn[start_idx:, start_idx:]

    # 找出最关注的10个位置
    # 1. 计算每一列（Key）被后续所有 Token 关注的平均强度
    # 排除掉最左侧的前 5 个 Sink Tokens，避免它们干扰排名
    col_means = sliced_attn[:, 5:].mean(axis=0)

    # 2. 找出平均注意力最高的 Top 10 个位置的索引
    # 注意要加上刚才偏移的 5
    top_k = 10
    top_indices = np.argsort(col_means)[-top_k:] + 5

    for idx in reversed(top_indices):
        token_str = clean_tokens[idx]
        importance_score = col_means[idx - 5]
        print(f"Index: {idx:4d} | Score: {importance_score:.4f} | Token: [{token_str}]")

    # 6. 绘制热力图
    plt.figure(figsize=(16, 14))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    sns.heatmap(
        sliced_attn,
        xticklabels=clean_tokens,
        yticklabels=clean_tokens,
        cmap='mako',  # 换一种冷色调，在文字密集时对比度更好
        vmax=0.05,
        cbar=True
    )
    plt.title(f"GLM-4.1V-Thinking (Layer {layer_idx}) - OCR & Reasoning Attention", fontsize=16)
    plt.xlabel("Key (Information Source)", fontsize=12)
    plt.ylabel("Query (Current Generation Step)", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        "text_attention_heatmap.pdf",  # 后缀改为 .pdf 或 .svg 即可自动保存为矢量图
        bbox_inches='tight'
    )
    plt.show()


########################################################################################################################
def draw_image_attn():
    # 1. 加载模型（千万别忘掉 eager）
    model_path = "local_GLM-4.1V_thinking"
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
    image_path = "test_case/visualization/clean_input_image.jpg"
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
                         "text clues you use. Before tackling the problem, make a plan and break it down into two to "
                         "four progressive subproblems. After presenting your reasoning, provide the final simple answer at last, "
                         "making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. "
                         "Question: <q>What is the average number of the genres of movies made by directors who have worked in drama genres?</q>"}
            ]
        }
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    # 3. 前向传播：生成并提取 Full Attention
    with torch.no_grad():
        generate_kwargs = {"max_new_tokens": 1024, "do_sample": False, "temperature": 0.0}
        generated_ids = model.generate(**inputs, **generate_kwargs)
        # generated_ids = apply_zoomtext_and_glc(model, prompt, inputs, processor, image)
        answer = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        print("Answer:", answer)
        full_input_ids = generated_ids[0].unsqueeze(0)

        forward_kwargs = {
            "input_ids": full_input_ids,
            "attention_mask": torch.ones_like(full_input_ids),
            "output_attentions": True,
            # 必须把原始的视觉输入完整带上
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw
        }

        outputs = model(**forward_kwargs)

    # 4. 获取注意力与 Token 映射
    attentions = outputs.attentions
    tokens = [processor.decode([tok]) for tok in full_input_ids[0]]

    # 【核心阶段 1：利用官方占位符精准定位图像 Token】
    image_start_in_attn = -1
    image_end_in_attn = -1

    # 直接遍历 tokens，寻找明确的起止符
    for i, token in enumerate(tokens):
        if token == '<|begin_of_image|>':
            image_start_in_attn = i + 1  # 真正的图像 token 从下一个开始
        elif token == '<|end_of_image|>':
            image_end_in_attn = i  # 图像 token 到此为止（不包含）
            break

    if image_start_in_attn == -1 or image_end_in_attn == -1:
        raise ValueError("未能通过 '<|begin_of_image|>' 和 '<|end_of_image|>' 定位图像")

    image_key_indices = list(range(image_start_in_attn, image_end_in_attn))
    num_image_tokens = len(image_key_indices)
    print(f"DEBUG: 提取的图像 Attention 范围: [{image_start_in_attn} : {image_end_in_attn}] (共 {num_image_tokens} 个 Token)")

    # 【核心阶段 2：获取正确的二维网格大小并重构】
    grid_thw = inputs["image_grid_thw"][0]
    orig_h, orig_w = grid_thw[1].item(), grid_thw[2].item()

    # 应用 GLM-4V 的 2x2 池化压缩比例
    vis_h = orig_h // 2
    vis_w = orig_w // 2
    expected_compressed_tokens = vis_h * vis_w

    print(f"DEBUG: 原始网格 H={orig_h}, W={orig_w}。压缩后网格 H={vis_h}, W={vis_w}。")
    print(f"DEBUG: 预期 Token 数={expected_compressed_tokens}, 实际截取数={num_image_tokens}")

    if expected_compressed_tokens != num_image_tokens:
        raise ValueError(f"维度不匹配！压缩后网格需 {expected_compressed_tokens} 个，但实际找到 {num_image_tokens} 个。")

    # 【核心阶段 3：提取交叉注意力矩阵 (Layer 20)】
    layer_idx = 20
    layer_attn = attentions[layer_idx][0]
    avg_attn = layer_attn.mean(dim=0).cpu().float().numpy()

    # 取最后 5 个生成的词 (通常是最终生成的答案) 作为 Query 去关注图像
    # seq_len = avg_attn.shape[0]
    # text_query_indices = list(range(image_end_in_attn, seq_len))

    # 对下面step进行注意力可视化
    # Question: <q>What is the average number of the genres of movies made by directors who have worked in drama genres?</q>
    text_query_indices = list(range(1338, 1364))

    # 对下面step进行注意力可视化
    # I need to look at each director's bar and check if there's an orange segment (since drama is orange, per the
    # legend). The legend says Drama is orange, so I need to find directors with orange bars.
    # text_query_indices = list(range(1420, 1462))

    # 对下面step进行注意力可视化
    # 1. Ethan Coen, Joel Coen: Has an orange segment (Drama). So Drama is present.
    # text_query_indices = list(range(1474, 1496))

    # 对下面step进行注意力可视化
    # 2. Lee Unkrich: Does he have Drama? Let's check. His bar has yellow (Animation), blue (Adventure), cyan (Comedy),
    # gray (Family), but orange? Wait, Lee Unkrich's bar: let's check. The bar for Lee Unkrich has segments: yellow
    # (Animation), blue (Adventure), cyan (Comedy), gray (Family), teal (Fantasy)? Wait, no, the colors: orange is
    # Drama. Let me check each director's bar for orange.
    # text_query_indices = list(range(1497, 1601))

    # 对下面step进行注意力可视化
    # - Thriller (pink, different shade)
    # text_query_indices = list(range(2195, 2202))

    # 对下面step进行注意力可视化
    # 3. Steven Spielberg: Has orange (Drama) → yes (since his bar has orange segment).
    # text_query_indices = list(range(2383, 2402))

    # 对下面step进行注意力可视化
    # 10. Roberto Benigni: Bar has orange (Drama) → yes.
    # text_query_indices = list(range(2698, 2719))

    # 使用 np.ix_ 提取
    cross_attn_matrix = avg_attn[np.ix_(text_query_indices, image_key_indices)]

    # 合并 Query 维度，得到纯粹的图像热力向量
    target_image_attn = cross_attn_matrix.mean(axis=0)

    # 【核心阶段 4：重构成二维矩阵】
    print(f"DEBUG: 准备 Reshape 的注意力向量长度: {len(target_image_attn)}")
    image_attn_2d = target_image_attn.reshape(vis_h, vis_w)

    # 【核心阶段 5：可视化叠加（内存直通版）】
    # 1. 绘制热力图并保存到内存缓冲区
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    sns.heatmap(
        image_attn_2d,
        cmap='jet',  # jet 的红蓝渐变在热力图叠加时最直观
        cbar=False,
        vmax=0.001
    )

    # 核心修改：使用 BytesIO 代替硬盘文件
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)  # 将内存指针拨回起点

    # 2. 从内存直接解码为 OpenCV 矩阵
    heatmap_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    heatmap_bgr = cv2.imdecode(heatmap_array, cv2.IMREAD_COLOR)

    if heatmap_bgr is None:
        raise ValueError("严重错误：从内存解码热力图失败！")

    # 3. 图像矩阵严格对齐
    image_np = np.array(image.convert('RGB'), dtype=np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 直接从真实的原图矩阵中读取高(H)和宽(W)
    h, w = image_bgr.shape[:2]

    # 强行将热力图 Resize 到绝对一致的尺寸
    heatmap_resize = cv2.resize(heatmap_bgr, (w, h))
    print(f"DEBUG: 原图矩阵 shape: {image_bgr.shape}, 热力图矩阵 shape: {heatmap_resize.shape}")

    # 4. 叠加并显示
    alpha = 0.5
    beta = 0.7
    combined_img = cv2.addWeighted(heatmap_resize, alpha, image_bgr, beta, 0)

    print("Generating VISUAL ATTENTION Map successfully!")
    cv2.imwrite('test_case/visualization/image_attention.jpg', combined_img)

    # 用 PIL 弹窗显示最终叠加图
    Image.fromarray(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)).show()


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
    s = hsv[:, :, 1]  # Saturation (0-255)
    v = hsv[:, :, 2]  # Value/Brightness (0-255)

    # 3. 像素级背景提取：低饱和度 (发白/发灰) 且 高明度 (不包含纯黑文字)
    # 注意：纯黑文字的 V 值通常很低 (< 100)；彩色柱状图的 S 值通常很高 (> 50)
    bg_pixel_mask = ((s < 35) & (v > 200)).astype(np.uint8)

    # 4. 形态学操作（可选）：轻微膨胀文字和色块，防止边缘被误判为背景
    kernel = np.ones((3, 3), np.uint8)
    # 反转掩码（让图文变1），膨胀图文，再反转回来
    fg_pixel_mask = 1 - bg_pixel_mask
    fg_pixel_mask = cv2.dilate(fg_pixel_mask, kernel, iterations=1)
    bg_pixel_mask = 1 - fg_pixel_mask

    # 5. Patch 级网格化 (Pooling)
    # 将高分辨率的像素 Mask 下采样为大模型的 Token 网格大小 (vis_h, vis_w)
    mask_2d = np.zeros((vis_h, vis_w), dtype=np.uint8)

    patch_h = h / vis_h
    patch_w = w / vis_w

    for i in range(vis_h):
        for j in range(vis_w):
            # 计算当前 Patch 在原图中的坐标范围
            y1, y2 = int(i * patch_h), int((i + 1) * patch_h)
            x1, x2 = int(j * patch_w), int((j + 1) * patch_w)

            # 提取该 Patch 内的所有像素掩码
            patch_pixels = bg_pixel_mask[y1:y2, x1:x2]

            # 6. 计算背景像素的纯度占比
            bg_ratio = np.sum(patch_pixels) / patch_pixels.size

            # 如果该方块内 95% 以上都是背景色，才将其判定为需要擦除的背景 Token
            if bg_ratio > 0.95:
                mask_2d[i, j] = 1
            else:
                mask_2d[i, j] = 0

    return mask_2d


if __name__ == '__main__':
    response_of_VLM()
    # draw_text_attn()
    # draw_image_attn()
