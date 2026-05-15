import base64
import io
import os
import re
from PIL import Image, ImageDraw
from openai import OpenAI


# ==========================================
# 1. 辅助函数定义
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
    """根据物理坐标裁剪图片，高画质放大并返回 Base64 字符串"""
    xmin, ymin, xmax, ymax = physical_bbox

    crop_xmin = max(0, xmin - padding)
    crop_ymin = max(0, ymin - padding)
    crop_xmax = min(img.width, xmax + padding)
    crop_ymax = min(img.height, ymax + padding)

    cropped_img = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

    if scale_factor != 1.0:
        new_width = int(cropped_img.width * scale_factor)
        new_height = int(cropped_img.height * scale_factor)
        magnified_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        magnified_img = cropped_img

    buffer = io.BytesIO()
    magnified_img.save(buffer, format="JPEG", quality=95)

    save_dir = "test_case/magnified"
    physical_bbox_str = ''.join([str(x) for x in physical_bbox])
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{physical_bbox_str}.jpg")
    magnified_img.save(file_path)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ==========================================
# 2. 初始化与预处理
# ==========================================
client = OpenAI(
    api_key="EMPTY",
    base_url="http://10.3.112.34:22240/v1",
)

local_image_path = "test_case/image17.jpg"
question = "如果买一提椰汁，两罐啤酒和一瓶葡萄酒，总共得到多少个独立包装的饮品？"

base64_image, orig_size, resized_size = resize_and_encode(local_image_path)
mime_type = "image/jpeg" if local_image_path.split('.')[-1].lower() != 'png' else "image/png"

# ==========================================
# 3. 第一阶段：宏观规划 (Task Planning)
# ==========================================
print("\n=== Stage 1：模型全局扫描并进行任务规划 ===")

conversation_history = [{
    'role': 'user',
    'content': [
        {
            'type': 'text',
            'text': f"You are an expert visual reasoning agent. We need to solve the following question: <q>{question}</q>\n"
                    "Before finding specific clues, please make a detailed step-by-step plan to break down this problem into progressive sub-problems. "
                    "Analyze which general areas or categories in the image you need to focus on. "
                    "CRITICAL: Do NOT calculate the final answer, and do NOT output any bounding box coordinates yet. Just output your plan."
        },
        {
            'type': 'image_url',
            'image_url': {'url': f"data:{mime_type};base64,{base64_image}"}
        }
    ]
}]

response_plan = client.chat.completions.create(
    model='Qwen3.5-9B',
    temperature=0.0,
    messages=conversation_history
)

plan_text = response_plan.choices[0].message.content
print(f"【规划输出】:\n{plan_text}\n")
conversation_history.append({'role': 'assistant', 'content': plan_text})

# ==========================================
# 4. 第二阶段：基于规划的视觉锚定 (Visual Grounding)
# ==========================================
print("=== Stage 2：根据规划提取线索坐标 ===")

conversation_history.append({
    'role': 'user',
    'content': [
        {
            'type': 'text',
            'text': "Now, execute your plan. Scan the specific regions you identified and find the visual clues (text or values) required to solve the sub-problems. "
                    "For each clue, output its bounding box coordinates normalized to the [0, 1000] range in the format [xmin, ymin, xmax, ymax]. "
                    "Still, do NOT output the final answer yet."
        }
    ]
})

response_grounding = client.chat.completions.create(
    model='Qwen3.5-9B',
    temperature=0.0,
    messages=conversation_history
)

grounding_text = response_grounding.choices[0].message.content
print(f"【坐标提取输出】:\n{grounding_text}\n")
conversation_history.append({'role': 'assistant', 'content': grounding_text})

# 提取并去重坐标
pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
matches = re.findall(pattern, grounding_text)

extracted_bboxes = []
seen = set()
for match in matches:
    bbox = tuple(int(val) for val in match)
    if bbox not in seen:
        seen.add(bbox)
        extracted_bboxes.append(list(bbox))

print(f"去重后提取到的原始坐标数量: {len(extracted_bboxes)} 个")

# ==========================================
# 5. 第三阶段：微观纠偏与最终推理 (Review & Correction)
# ==========================================
if extracted_bboxes:
    print("\n=== Stage 3：局部高清复审、思维链纠偏与最终解答 ===")

    original_img = Image.open(local_image_path).convert('RGB')
    clean_img = original_img.copy()
    draw = ImageDraw.Draw(original_img)

    orig_w, orig_h = orig_size

    round_3_content = [
        {
            'type': 'text',
            'text': "Here are the high-resolution cropped images of the regions you just identified based on your plan. "
                    "1. First, list the precise text or values visible in these cropped details.\n"
                    "2. Second, reflect on your initial plan and findings. Correct any hallucinations, missed items, or logical errors from your previous steps.\n"
                    f"3. Finally, step-by-step, answer the original question: <q>{question}</q>. Enclose the final answer within the tags <a> and </a> like this: <a>your answer</a>."
        }
    ]

    print("正在裁剪并打包高清局部图...")
    for bbox in extracted_bboxes:
        xmin, ymin, xmax, ymax = bbox

        real_xmin = int((xmin / 1000.0) * orig_w)
        real_ymin = int((ymin / 1000.0) * orig_h)
        real_xmax = int((xmax / 1000.0) * orig_w)
        real_ymax = int((ymax / 1000.0) * orig_h)
        physical_bbox = [real_xmin, real_ymin, real_xmax, real_ymax]

        cropped_base64 = crop_and_encode(clean_img, physical_bbox, padding=60)

        round_3_content.append({
            'type': 'image_url',
            'image_url': {'url': f"data:image/jpeg;base64,{cropped_base64}"}
        })

        draw.rectangle(physical_bbox, outline="#00FF00", width=4)

    conversation_history.append({'role': 'user', 'content': round_3_content})

    print("模型正在结合高清切图进行思维链自我纠偏与最终推理...")
    response_final = client.chat.completions.create(
        model='Qwen3.5-9B',
        temperature=0.0,
        messages=conversation_history
    )

    final_answer_text = response_final.choices[0].message.content
    print(f"\n【最终思维链与解答】:\n{final_answer_text}")

    output_path = "test_case/result_image_multi_turn.jpg"
    original_img.save(output_path)
    print(f"\n全局绘制成功！带框的图片已保存至: {output_path}")

else:
    print("\n未提取到任何坐标，流水线终止。")