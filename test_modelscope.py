import os
import base64
import io
from openai import OpenAI
import re
from PIL import Image, ImageDraw


# 1. 定义一个函数，用于将本地图片读取并转换为 Base64 编码
def encode_and_resize_image(image_path, max_size=2048):
    # 打开本地图片
    with Image.open(image_path) as img:
        orig_size = img.size
        # 如果宽或高超过了限制，则进行等比例缩放
        if img.width > max_size or img.height > max_size:
            # thumbnail 方法会自动保持长宽比例进行缩放
            img.thumbnail((max_size, max_size))
            print(f"[resize] 图片过大，已自动缩放为: {img.width}x{img.height}")

        resized_size = img.size  # (width, height)

        # 将图片保存到内存中的字节缓冲区
        buffer = io.BytesIO()
        # 为了兼容性并减小传输体积，统一转换为 RGB 模式并以 JPEG 格式输出
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG")

        # 返回 Base64 编码
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
    magnified_img.show()
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# client = OpenAI(
#     base_url='https://api-inference.modelscope.cn/v1',
#     api_key='ms-7ee485f4-550b-49f8-906a-216377ad5562',  # ModelScope Token
# )

client = OpenAI(
    api_key="",
    base_url="http://10.3.112.34:22240/v1",
)


# Question0: What is the average number of the genres of movies made by directors who have worked in drama genres?  # Answer: 3.625
# Question1: 某人购买了所有劲爆价的商品各一份，其中脉动饮料买了3瓶，那么他总共会得到多少瓶饮料？  # Answer: 58
# Question2: How many Serviced K users which have meters value less than 5?  # Answer: 12
# Question3: 某人A买了一个财源滚滚金砂旺旺，他的朋友买了双鱼板结40号，两个灯笼福串和一个1.6m的精品鱼版对联，他最少要付多少钱？  # Answer: 61.23元
# Question4: How many are somewhat confident that Donald Trump can manage the executive branch effectively?  # Answer: 24%
# Question5: 历年长庆油田新增探明储量的数值的标准差是多少？  # Answer: 1.0
# Question6: Answer the question in the image. If it is a multiple-choice question, please provide the option you think is correct. If it is free-form, please provide your answer.  # Answer: 150
# Question7: 回答图片中的问题。如果是选择题，请提供你认为正确的选项。如果是简答题，请提供你的答案。  # Answer: D
# Question8: Answer the question in the image. If it is a multiple-choice question, please provide the option you think is correct. If it is free-form, please provide your answer.  # Answer: C
# Question9: 我家距离场馆有24km，其中有4km的路程限速20km/h，其余路段限速30km/h，为了能在停止检票前半小时入场，我至少需要在那个时刻出发？  # Answer: 19:38
# Question10: If a customer wants to add the service charge to the bill, how much will it cost?  # Answer: 50.46
# Question11: 第2小点对应的那栏的数字之和是多少？  # Answer: 22555.3
# Question12: On the left chart, how many models have ever reached a Top-1 Accuracy over 80% on the K400 dataset?  # Answer: 6

# 2. 指定你的本地图片路径 (请替换为你实际的图片路径，例如 "C:/images/test.jpg" 或 "./test.png")
local_image_path = "test_case/image14.jpg"
detail_image_path = "test_case/detail13.png"

original_img = Image.open(local_image_path).convert('RGB')
orig_size = original_img.size
orig_w, orig_h = orig_size
# 3. 获取 Base64 编码的字符串
bbox = [378, 530, 536, 550]
xmin, ymin, xmax, ymax = bbox
real_xmin = int((xmin / 1000.0) * orig_w)
real_ymin = int((ymin / 1000.0) * orig_h)
real_xmax = int((xmax / 1000.0) * orig_w)
real_ymax = int((ymax / 1000.0) * orig_h)
physical_bbox = [real_xmin, real_ymin, real_xmax, real_ymax]
base64_image = crop_and_encode(original_img, physical_bbox)


# base64_image, orig_size, resized_size = encode_and_resize_image(local_image_path)
# base64_image_detail, _, _ = encode_and_resize_image(detail_image_path)

# 判断图片类型以生成正确的 MIME type (可选，通常传 jpeg 兼容性最好)
image_format = local_image_path.split('.')[-1].lower()
image_detail_format = detail_image_path.split('.')[-1].lower()
if image_format == 'png':
    mime_type = "image/png"
else:
    mime_type = "image/jpeg"

if image_detail_format == 'png':
    mime_type_detail = "image/png"
else:
    mime_type_detail = "image/jpeg"

    "Please pay close attention to the textual information in the image, as well as the key elements "
    "specified in the question, such as the objects, relationships, and constraints. "
    "Then answer the following question within the tags <q> and </q> step-by-step "
    "according to the textual content, explaining your reasoning process clearly at each stage and the "
    "text clues you use. Before tackling the problem, make a plan and break it down into two to "
    "four progressive subproblems. After presenting your reasoning, provide the final simple answer at last, "
    "making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. "
    "Question: <q>How many numbers are preceded by a '$'?"

    "Answer the following question within the tags <q> and </q> step-by-step. "
    "After presenting your reasoning, provide the final simple answer at last, "
    "making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. "
    "Question: <q>2021年的不同业务的同比增长率的方差是多少？</q>"

    "Answer the following question within the tags <q> and </q> step-by-step. "
    "For each text clue or value you extract from the image, you must provide its bounding box coordinates normalized to the [0, 1000] range in the format [xmin, ymin, xmax, ymax]. "
    "After presenting your reasoning, provide the final simple answer at last, "
    "making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. "
    "Question: <q>How many are somewhat confident that Donald Trump can manage the executive branch effectively?</q>"

response = client.chat.completions.create( # type: ignore
    model='Qwen3.5-9B',
    temperature=0.0,
    messages=[{ # type: ignore
        'role':
            'user',
        'content': [{
            'type': 'text',
            'text': "You are an expert visual assistant. You are asked to scan the image and answer the question: <q>What is the total price of the foods whose names end with the letter 'r'?"
                    "</q> step-by-step. Find the clues in the image which are related to the question. For each text clue or value you extract, output its bounding box coordinates normalized to the [0, 1000] range in the format [xmin, ymin, xmax, ymax]. "
                    "Do not calculate the final total price yet, just list the items, prices, and their coordinates.",
        },
        {
            'type': 'image_url',
            'image_url': {
                # 4. 将原本的 http URL 替换为 Base64 Data URL 格式
                'url': f"data:{mime_type};base64,{base64_image}",
            },
        },
        # {
        #     'type': 'image_url',
        #     'image_url': {
        #         # 4. 将原本的 http URL 替换为 Base64 Data URL 格式
        #         'url': f"data:{mime_type_detail};base64,{base64_image_detail}",
        #     },
        # }
        ],
    }],
    # stream=True
)

# huawei集群qwen3.5-27B接口输出
final_answer_text = response.choices[0].message.content
print(final_answer_text)

# # modelscope Glm4.6V接口输出
# for chunk in response:
#     if chunk.choices:
#         print(chunk.choices[0].delta.content, end='', flush=True)

# model scope qwen3.5-27B接口输出
# final_answer_text = ""
# done_reasoning = False
# for chunk in response:
#     if chunk.choices:
#         reasoning_chunk = chunk.choices[0].delta.reasoning_content
#         answer_chunk = chunk.choices[0].delta.content
#         if reasoning_chunk != '':
#             print(reasoning_chunk, end='', flush=True)
#             final_answer_text += reasoning_chunk
#         elif answer_chunk != '':
#             if not done_reasoning:
#                 print('\n\n === Final Answer ===\n')
#                 done_reasoning = True
#             print(answer_chunk, end='', flush=True)
#             final_answer_text += answer_chunk

print("\n\n=== 提取与可视化阶段 ===")

# 2. 使用正则表达式提取坐标
# 这个正则匹配形如 [ymin, xmin, ymax, xmax] 的格式，允许数字前后有空格
pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
matches = re.findall(pattern, final_answer_text)

# 将匹配到的字符串元组转换为整数列表
extracted_bboxes = [[int(val) for val in match] for match in matches]
unique_bboxes = []
seen = set()
for bbox in extracted_bboxes:
    # 列表不能作为 set 的 key，必须先转成 tuple
    bbox_tuple = tuple(bbox)
    if bbox_tuple not in seen:
        seen.add(bbox_tuple)
        unique_bboxes.append(bbox)

# 将去重后的结果覆盖回原变量
extracted_bboxes = unique_bboxes
print(f"去重提取到的原始坐标 (0-1000格式): {extracted_bboxes}")

# 3. 在原图上可视化框
if extracted_bboxes:
    try:
        # 【关键修改 1】：打开原图，并立刻复制一份干净的副本
        original_img = Image.open(local_image_path).convert('RGB')
        clean_img = original_img.copy()  # 专门用于无损裁剪

        # 在 original_img 上进行画框操作
        draw = ImageDraw.Draw(original_img)

        orig_w, orig_h = orig_size
        # resized_w, resized_h 已经不再需要了

        # 3 & 4. 强制按 [0-1000] 归一化坐标系进行精确映射并画框
        for i, bbox in enumerate(extracted_bboxes):
            xmin, ymin, xmax, ymax = bbox

            # 映射到物理坐标
            real_xmin = int((xmin / 1000.0) * orig_w)
            real_ymin = int((ymin / 1000.0) * orig_h)
            real_xmax = int((xmax / 1000.0) * orig_w)
            real_ymax = int((ymax / 1000.0) * orig_h)

            physical_bbox = [real_xmin, real_ymin, real_xmax, real_ymax]
            # print(f"映射后原图坐标: {physical_bbox}")

            # 【关键修改 2】：调用复审机制 (注意传入的是 clean_img)
            # 为了防止一次性请求太多次（你提取了12个框），你可以先限制只审查前 3 个框测试一下

            # 画出边框 (画在 original_img 上)
            draw.rectangle(
                physical_bbox,
                outline="#00FF00",  # 亮绿色
                width=4
            )

        output_path = "test_case/result_image.jpg"
        original_img.save(output_path)
        print(f"\n✅ 全局绘制成功！带框的图片已保存至: {output_path}")

        original_img.show()

    except Exception as e:
        print(f"图片处理失败: {e}")
else:
    print("未提取到坐标。")