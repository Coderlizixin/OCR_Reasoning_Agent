import os
import pandas as pd
import base64
import io
import re
from PIL import Image, ImageDraw
from openai import OpenAI

# ==========================================
# 1. 全局配置与客户端初始化
# ==========================================
client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key='ms-7ee485f4-550b-49f8-906a-216377ad5562',  # ModelScope Token
)

# client = OpenAI(
#     api_key="",
#     base_url="http://10.3.112.34:22510/v1",
# )

# ==========================================
# 2. 内存图像处理工具函数
# ==========================================
def process_base64_to_images(base64_str, max_size=8192):
    """
    将原始 Base64 解码为 PIL Image 对象，并生成一张用于全局视角的缩放版 Base64。
    """
    img_data = base64.b64decode(base64_str)
    original_img = Image.open(io.BytesIO(img_data))

    if original_img.mode in ("RGBA", "P"):
        original_img = original_img.convert("RGB")

    orig_w, orig_h = original_img.size

    global_img = original_img.copy()
    if max(global_img.size) > max_size:
        global_img.thumbnail((max_size, max_size))

    buffer = io.BytesIO()
    global_img.save(buffer, format="JPEG")
    global_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return original_img, (orig_w, orig_h), global_base64


def crop_and_encode(img, physical_bbox, padding=60, scale_factor=3.0):
    """
    根据物理坐标在内存中裁剪图片，带边界保护、高画质放大，返回 Base64。
    """
    xmin, ymin, xmax, ymax = physical_bbox

    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    if xmin == xmax:
        xmax += 1
    if ymin == ymax:
        ymax += 1

    crop_xmin = max(0, xmin - padding)
    crop_ymin = max(0, ymin - padding)
    crop_xmax = min(img.width, xmax + padding)
    crop_ymax = min(img.height, ymax + padding)

    cropped_img = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

    if scale_factor != 1.0:
        new_width = max(1, int(cropped_img.width * scale_factor))
        new_height = max(1, int(cropped_img.height * scale_factor))
        cropped_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    cropped_img.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ==========================================
# 3. 边界框绘制函数
# ==========================================
def draw_bboxes_on_image(image, bboxes, orig_w, orig_h, output_path):
    """
    在图片上绘制边界框并保存。

    Args:
        image: PIL Image 对象
        bboxes: 归一化到 [0, 1000] 范围的坐标列表 [[xmin, ymin, xmax, ymax], ...]
        orig_w: 原始图片宽度 (像素)
        orig_h: 原始图片高度 (像素)
        output_path: 输出图片保存路径
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    colors = [
        "#00FF00", "#FF0000", "#0000FF", "#FFA500", "#FF00FF",
        "#00FFFF", "#FFFF00", "#800080", "#008000", "#FF69B4"
    ]

    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox

        real_xmin = int((xmin / 1000.0) * orig_w)
        real_ymin = int((ymin / 1000.0) * orig_h)
        real_xmax = int((xmax / 1000.0) * orig_w)
        real_ymax = int((ymax / 1000.0) * orig_h)

        color = colors[i % len(colors)]
        draw.rectangle(
            [real_xmin, real_ymin, real_xmax, real_ymax],
            outline=color,
            width=5
        )
        # 在框的左上角标注序号
        draw.text(
            (real_xmin + 6, real_ymin + 6),
            str(i + 1),
            fill=color
        )

    annotated.save(output_path, quality=95)
    return annotated


# ==========================================
# 4. 单条数据处理函数
# ==========================================
def process_single_row(index, row, output_dir):
    """
    将单条 OCR-Reasoning 数据发送给 Qwen3.5-122B-A10B，
    让模型给出推理所需视觉元素的 bounding box，并在图上绘制出来。

    返回结果字典。
    """
    item_id = row.get('index', index)
    question = str(row.get('question', ''))
    raw_img_str = row.get('image', None)
    ground_truth = str(row.get('answer', ''))
    reasoning = str(row.get('reasoning', ''))

    if pd.isna(raw_img_str):
        return {"id": item_id, "status": "error", "error_msg": "图片缺失"}

    try:
        original_img, (orig_w, orig_h), global_base64 = process_base64_to_images(raw_img_str)

        # 构造单轮提示：图片 + 问题 + 答案 + 推理过程 → 要求输出坐标
        prompt_text = f"""You are an expert visual reasoning assistant. I will provide you with an image, a question about the image, the correct answer, and the step-by-step reasoning used to arrive at that answer.
=== TASK ===
Identify ALL visual elements (text blocks, numbers, chart regions, labels, data points, etc.) in the image that are NECESSARY to solve the question. For each such element, output its bounding box coordinates.
=== QUESTION ===
{question}

=== CORRECT ANSWER ===
{ground_truth}

=== REASONING PROCESS ===
{reasoning}

=== OUTPUT FORMAT ===
For each visual element, output one line in this exact format:
Element N: <brief description> - [xmin, ymin, xmax, ymax]

All coordinates MUST be normalized to the [0, 1000] range. Example:
Element 1: chart title "Annual Sales 2023" - [120, 50, 880, 120]
Element 2: Y-axis label "Revenue (millions USD)" - [30, 100, 80, 500]

Only output bounding boxes for elements that are CRITICAL for answering the question."""

        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt_text},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{global_base64}"},
                 'min_pixels': 2500 * 32 * 32,
                 'max_pixels': 6000 * 32 * 32,
                 }
            ]
        }]

        response = client.chat.completions.create(
            # model='Qwen/Qwen3.5-122B-A10B',
            model='Qwen/Qwen3.5-397B-A17B',
            temperature=0.0,
            max_tokens=4096,
            messages=messages
        )
        response_text = response.choices[0].message.content

        # 解析 bounding box 坐标
        pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        matches = re.findall(pattern, response_text)
        extracted_bboxes = []
        seen = set()
        for match in matches:
            bbox_tuple = tuple(int(val) for val in match)
            if bbox_tuple not in seen:
                seen.add(bbox_tuple)
                extracted_bboxes.append(list(bbox_tuple))

        # 绘制边界框
        output_image_path = None
        if extracted_bboxes:
            safe_id = str(item_id).replace('/', '_').replace('\\', '_')
            output_filename = f"annotated_{safe_id}.jpg"
            output_image_path = os.path.join(output_dir, output_filename)
            draw_bboxes_on_image(
                image=original_img,
                bboxes=extracted_bboxes,
                orig_w=orig_w,
                orig_h=orig_h,
                output_path=output_image_path
            )

        return {
            "id": item_id,
            "question": question,
            "ground_truth": ground_truth,
            "reasoning": reasoning,
            "model_response": response_text,
            "extracted_bboxes": extracted_bboxes,
            "bbox_count": len(extracted_bboxes),
            "annotated_image": output_image_path,
            "status": "success"
        }

    except Exception as e:
        return {"id": item_id, "status": "error", "error_msg": str(e)}


# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    tsv_path = r"C:\Users\19639\LMUData\OCR_Reasoning.tsv"
    output_dir = r"test_case/bbox_annotations"
    os.makedirs(output_dir, exist_ok=True)

    try:
        print("正在加载数据集...")
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"成功加载，共 {len(df)} 条数据。取前 5 条进行处理。\n")

        subset = df.head(1)
        results = []

        for idx, row in subset.iterrows():
            item_id = row.get('index', idx)
            print(f"--- 处理第 {idx + 1} 条 (ID: {item_id}) ---")
            result = process_single_row(idx, row, output_dir)
            results.append(result)

            if result['status'] == 'success':
                print(f"  提取到 {result['bbox_count']} 个边界框")
                if result['annotated_image']:
                    print(f"  带框图片已保存: {result['annotated_image']}")
            else:
                print(f"  错误: {result['error_msg']}")

        # 保存结果
        results_df = pd.DataFrame(results)
        results_output = os.path.join(output_dir, "bbox_results.tsv")
        results_df.to_csv(results_output, sep='\t', index=False, encoding='utf-8')
        print(f"\n结果已保存至: {results_output}")

    except FileNotFoundError:
        print(f"未找到文件: {tsv_path}")
