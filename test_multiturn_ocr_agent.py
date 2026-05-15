import base64
import io
import os
import re
from PIL import Image, ImageDraw
from openai import OpenAI

# ==========================================
# 1. 辅助函数定义：图像缩放编码函数和图像裁切放大编码函数
# ==========================================

def resize_and_encode(image_path, max_size=2048):
    """读取本地图片，按比例缩放并转换为 Base64 编码"""
    with Image.open(image_path) as img:
        orig_size = img.size
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))
            print(f"[resize] 图片过大，已自动缩放为: {img.width}x{img.height}")

        resized_size = img.size
        buffer = io.BytesIO()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), orig_size, resized_size


def crop_and_encode(img, physical_bbox, padding=60, scale_factor=3.0):
    """根据物理坐标裁剪图片，并返回 Base64 字符串"""
    xmin, ymin, xmax, ymax = physical_bbox

    crop_xmin = max(0, xmin - padding)
    crop_ymin = max(0, ymin - padding)
    crop_xmax = min(img.width, xmax + padding)
    crop_ymax = min(img.height, ymax + padding)

    cropped_img = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

    if scale_factor != 1.0:
        new_width = int(cropped_img.width * scale_factor)
        new_height = int(cropped_img.height * scale_factor)

        # Image.Resampling.LANCZOS 是最高质量的重采样滤波器，非常适合放大文字，不会出现严重马赛克
        magnified_img = cropped_img.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
    else:
        magnified_img = cropped_img

    # print(f"🔍 原始小图尺寸: {cropped_img.width}x{cropped_img.height} -> 放大后尺寸: {magnified_img.width}x{magnified_img.height}")

    buffer = io.BytesIO()
    # 质量可以适当调高，确保放大后的细节不被 JPEG 压缩破坏
    magnified_img.save(buffer, format="JPEG", quality=95)
    save_dir = "test_case/magnified"
    physical_bbox_str = ''.join([str(x) for x in physical_bbox])
    # 2. Create the directory if it doesn't exist (exist_ok=True prevents errors if it's already there)
    os.makedirs(save_dir, exist_ok=True)

    # 3. Construct the full file path safely
    # Assuming physical_bbox_str is defined earlier in your function
    file_path = os.path.join(save_dir, f"{physical_bbox_str}.jpg")

    # 4. Save the image
    magnified_img.save(file_path)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ==========================================
# 2. 初始化与预处理
# ==========================================

# 初始化 OpenAI 客户端 (请替换为你实际的本地/内网服务地址)
client = OpenAI(
    api_key="",
    base_url="http://10.3.112.34:22240/v1",
)

local_image_path = "test_case/image14.jpg"
question = "What is the total price of the foods whose names end with the letter 'r'?"

# 获取 Base64 编码和尺寸
base64_image, orig_size, resized_size = resize_and_encode(local_image_path)
mime_type = "image/jpeg" if local_image_path.split('.')[-1].lower() != 'png' else "image/png"

# ==========================================
# 3. 智能体执行：第一轮对话 (全局扫描与坐标定位)
# ==========================================
print("\n=== 第一轮：模型全局扫描并提取坐标 ===")

# 维护全局对话历史
conversation_history = [{
    'role': 'user',
    'content': [
        {
            'type': 'text',
            'text': "You are an expert visual assistant. You are asked to scan the image and answer the question: <q>" + question +
                    "</q> step-by-step. Find the clues in the image which are related to the question. "
                    "For each text clue or value you extract, output its bounding box coordinates normalized to the [0, 1000] range in the format [xmin, ymin, xmax, ymax]. "
                    "Do not answer the question yet, just list the items as clues."
        },
        {
            'type': 'image_url',
            'image_url': {
                'url': f"data:{mime_type};base64,{base64_image}",
            },
        }
    ]
}]

response_round_1 = client.chat.completions.create(
    model='Qwen3.5-9B',  # 请确保内网部署的这个模型支持视觉能力
    temperature=0.0,
    messages=conversation_history
)

first_round_text = response_round_1.choices[0].message.content
print(f"第一轮输出：\n{first_round_text}\n")

# 将模型第一轮的回答追加到对话历史中，保持上下文不断裂！
conversation_history.append({
    'role': 'assistant',
    'content': first_round_text
})

# ==========================================
# 4. 坐标提取与去重
# ==========================================
print("=== 坐标提取与处理阶段 ===")

pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
matches = re.findall(pattern, first_round_text)

extracted_bboxes = [[int(val) for val in match] for match in matches]
unique_bboxes = []
seen = set()
for bbox in extracted_bboxes:
    bbox_tuple = tuple(bbox)
    if bbox_tuple not in seen:
        seen.add(bbox_tuple)
        unique_bboxes.append(bbox)

extracted_bboxes = unique_bboxes
print(f"去重后提取到的原始坐标数量: {len(extracted_bboxes)} 个")
print(f"坐标列表: {extracted_bboxes}")

# ==========================================
# 5. 智能体执行：第二轮对话 (局部高清复审与最终推理)
# ==========================================
if extracted_bboxes:
    print("\n=== 第二轮：局部高清复审与最终解答 ===")

    original_img = Image.open(local_image_path).convert('RGB')
    clean_img = original_img.copy()  # 专门用于无损裁剪
    draw = ImageDraw.Draw(original_img)  # 用于最后的可视化画框

    orig_w, orig_h = orig_size

    # 准备第二轮用户消息的 Content 数组
    round_2_content = [
        {
            'type': 'text',
            'text': "Here are the high-resolution cropped images of the regions you just identified. "
                    "First, you must list the details in the cropped images. "
                    "Then, review these details carefully and correct the errors in the first round. "
                    "Finally, step-by-step, answer the original question: <q>" + question +
                    "</q>. Make sure to enclose the final total price within the tags <a> and </a> like this: <a>your answer</a>."
        }
    ]

    print("正在裁剪并打包高清局部图...")
    for i, bbox in enumerate(extracted_bboxes):
        xmin, ymin, xmax, ymax = bbox

        # 映射到物理坐标
        real_xmin = int((xmin / 1000.0) * orig_w)
        real_ymin = int((ymin / 1000.0) * orig_h)
        real_xmax = int((xmax / 1000.0) * orig_w)
        real_ymax = int((ymax / 1000.0) * orig_h)
        physical_bbox = [real_xmin, real_ymin, real_xmax, real_ymax]

        # 裁剪并编码为 Base64
        cropped_base64 = crop_and_encode(clean_img, physical_bbox, padding=60)

        # 将裁剪出的小图追加到第二轮的 content 列表中
        round_2_content.append({
            'type': 'image_url',
            'image_url': {
                'url': f"data:image/jpeg;base64,{cropped_base64}"
            }
        })

        # 在原图上画框 (用于最终保存查看)
        draw.rectangle(physical_bbox, outline="#00FF00", width=4)

    # 将打包好的第二轮问题（文字+多张裁剪小图）加入对话历史
    conversation_history.append({
        'role': 'user',
        'content': round_2_content
    })

    # 发起第二轮请求
    print("模型正在结合局部高清细节进行最终推理...")
    response_round_2 = client.chat.completions.create(
        model='Qwen3.5-9B',
        temperature=0.0,
        messages=conversation_history
    )

    final_answer_text = response_round_2.choices[0].message.content
    print(f"\n【最终解答】:\n{final_answer_text}")

    # ==========================================
    # 6. 保存可视化结果
    # ==========================================
    output_path = "test_case/result_image_multi_turn.jpg"
    original_img.save(output_path)
    print(f"\n✅ 全局绘制成功！带框的图片已保存至: {output_path}")
    original_img.show()

else:
    print("\n⚠️ 未提取到任何坐标，跳过第二轮复审。")