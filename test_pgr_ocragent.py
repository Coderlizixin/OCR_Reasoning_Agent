"""
独立测试脚本：对单个实例运行 ThreeStageOCRAgent 三阶段流水线。
与 vlmeval/vlm/pgr_ocragent.py 逻辑一致，但不依赖 VLMEvalKit 框架，
可直接通过 `python test_pgr_ocragent.py` 运行。

三阶段设计：
  Stage 1 (Planning) : 9B @ port 22240 — 原子化可执行计划
  Stage 2 (Grounding): 122B @ port 22510 — 带语义标签的坐标定位
  Stage 3 (Reasoning): 9B @ port 22630 — 结合标签裁剪图 + 原图缩略图进行推理
"""
import base64
import io
import json
import os
import re
import sys
import time as _time
from PIL import Image, ImageDraw
from openai import OpenAI


# ==========================================
# 0. 配置：修改这里的图片路径和问题即可测试不同实例
# ==========================================
LOCAL_IMAGE_PATH = "test_case/test_images/image14.jpg"
QUESTION = "What is the total price of the foods whose names end with the letter 'r'?"

# API 端点配置（与 pgr_ocragent.py 保持一致）
STAGE1_BASE_URL = "http://10.3.112.34:22240/v1"   # Stage 1: Qwen3.5-9B 规划
STAGE2_BASE_URL = "http://10.3.112.34:22510/v1"   # Stage 2: Qwen3.5-122B-A10B 定位
STAGE3_BASE_URL = "http://10.3.112.34:22630/v1"   # Stage 3: Qwen3.5-9B 推理

STAGE1_MODEL = "Qwen3.5-9B"
STAGE2_MODEL = "Qwen3.5-122B-A10B"
STAGE3_MODEL = "Qwen3.5-9B"

# 裁剪参数
PADDING = 60
SCALE_FACTOR = 3.0
MAX_CROP_EDGE = 1024      # 裁剪子图最长边上限（控制 visual token）
MAX_ORIG_EDGE = 1024      # Stage 3 附带原图缩略图的最长边
MAX_CROPS = 12             # 最多裁剪区域数

# 输出目录
OUTPUT_DIR = "test_case/pgr_test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 1. 辅助函数
# ==========================================

def resize_and_encode(image_path, max_size=2048):
    """读取本地图片，按比例缩放并转换为 Base64。返回 (base64_str, orig_w, orig_h)。"""
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))
            print(f"[resize] 图片过大，已自动缩放为: {img.width}x{img.height}")
        buffer = io.BytesIO()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), orig_w, orig_h


def crop_and_encode(image_path, physical_bbox, padding=PADDING, scale_factor=SCALE_FACTOR,
                    max_edge=MAX_CROP_EDGE):
    """根据物理坐标裁剪、放大、限制最长边，返回 Base64 字符串（纯内存操作）。"""
    with Image.open(image_path).convert('RGB') as img:
        xmin, ymin, xmax, ymax = physical_bbox

        # 坐标纠正
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin == xmax:
            xmax += 1
        if ymin == ymax:
            ymax += 1

        # 带 padding 裁剪
        c_xmin = max(0, xmin - padding)
        c_ymin = max(0, ymin - padding)
        c_xmax = min(img.width, xmax + padding)
        c_ymax = min(img.height, ymax + padding)

        cropped = img.crop((c_xmin, c_ymin, c_xmax, c_ymax))

        # 放大
        if scale_factor != 1.0:
            new_w = int(cropped.width * scale_factor)
            new_h = int(cropped.height * scale_factor)
            cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 限制最长边
        if max_edge is not None:
            w, h = cropped.size
            _max = max(w, h)
            if _max > max_edge:
                ratio = max_edge / _max
                cropped = cropped.resize((int(w * ratio), int(h * ratio)),
                                         Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        cropped.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


def encode_pil_to_base64(pil_image, max_edge=MAX_ORIG_EDGE):
    """将 PIL 图像缩放（如需要）后编码为 Base64 JPEG。"""
    w, h = pil_image.size
    _max = max(w, h)
    if max_edge is not None and _max > max_edge:
        ratio = max_edge / _max
        pil_image = pil_image.resize((int(w * ratio), int(h * ratio)),
                                     Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def build_image_content(b64_string):
    """构造 OpenAI API 格式的 image_url 内容块。"""
    return {
        'type': 'image_url',
        'image_url': {'url': f"data:image/jpeg;base64,{b64_string}"}
    }


def build_text_content(text):
    """构造 OpenAI API 格式的 text 内容块。"""
    return {'type': 'text', 'text': text}


def log(msg):
    """带时间戳的日志输出。"""
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}")


# ==========================================
# 2. 初始化客户端
# ==========================================

stage1_client = OpenAI(api_key="EMPTY", base_url=STAGE1_BASE_URL)
grounding_client = OpenAI(api_key="ms-EMPTY", base_url=STAGE2_BASE_URL)
stage3_client = OpenAI(api_key="EMPTY", base_url=STAGE3_BASE_URL)

log(f"Stage 1 client: {STAGE1_BASE_URL}")
log(f"Stage 2 client: {STAGE2_BASE_URL}")
log(f"Stage 3 client: {STAGE3_BASE_URL}")


# ==========================================
# 3. 预处理：图片编码
# ==========================================

base64_image, orig_w, orig_h = resize_and_encode(LOCAL_IMAGE_PATH, max_size=2048)
mime_type = "image/jpeg" if LOCAL_IMAGE_PATH.split('.')[-1].lower() != 'png' else "image/png"

log(f"图片加载完成: {LOCAL_IMAGE_PATH}")
log(f"原始尺寸: {orig_w}x{orig_h}")
log(f"问题: {QUESTION}")

# ==========================================
# 4. Stage 1: 原子化规划 (Atomic Planning)
# ==========================================
_t1_start = _time.time()
log("=" * 60)
log("STAGE 1: 原子化规划 (Atomic Planning)")

prompt_s1 = (
    f"You are an expert visual reasoning agent. We need to solve the following question: "
    f"<q>{QUESTION}</q>\n\n"
    "Create a detailed, atomic execution plan. Each step must be so specific that even a basic OCR model "
    "or entry-level grounding model can execute it without any reasoning.\n\n"
    "RULES for your plan:\n"
    "- Each step must target exactly ONE visual element (a single number, label, word, or shape)\n"
    "- For each step, describe the element precisely: what it looks like, where it is semantically located "
    "(e.g., 'the number printed next to \"青菜组合\" in the vegetable section of the menu', "
    "NOT 'prices of all vegetables')\n"
    "- Include the expected data type for each step (number, text, option letter, etc.)\n"
    "- Steps should be ordered logically: gather all raw data first, then compute\n\n"
    "OUTPUT FORMAT — a numbered list. Each line must follow this exact template:\n"
    'Step N: [action] — locate "<exact element description>" (expected: <data type>)\n\n'
    "Example:\n"
    'Step 1: Read the price — locate "the number next to \\"青菜组合\\" in the vegetable section of the menu" (expected: number)\n'
    'Step 2: Read the price — locate "the number next to \\"豆制品拼盘\\" in the soy products section of the menu" (expected: number)\n'
    'Step 3: Compute total — add the two prices from Step 1 and Step 2 (expected: number)\n\n'
    "CRITICAL: Do NOT calculate any final answer. Do NOT output bounding box coordinates. "
    "Only output the atomic plan as a numbered list following the format above."
)

s1_response = stage1_client.chat.completions.create(
    model=STAGE1_MODEL,
    temperature=0.0,
    max_tokens=6144,
    messages=[{
        'role': 'user',
        'content': [build_text_content(prompt_s1), build_image_content(base64_image)]
    }]
)

plan_text = s1_response.choices[0].message.content
_t1_elapsed = _time.time() - _t1_start
log(f"Stage 1 完成 ({_t1_elapsed:.0f}s)")
print(f"\n{'─' * 40}\n【规划输出】:\n{plan_text}\n{'─' * 40}\n")


# ==========================================
# 5. Stage 2: 带语义标签的视觉定位 (Labeled Grounding)
# ==========================================
_t2_start = _time.time()
log("STAGE 2: 带语义标签的视觉定位 (Labeled Grounding)")

prompt_s2 = (
    "Now, execute your plan step by step. For each plan step, locate the exact visual element described "
    "and output BOTH a semantic label AND the bounding box coordinates.\n\n"
    "CRITICAL SYSTEM INSTRUCTIONS - YOU MUST FOLLOW THESE STRICTLY:\n"
    "1. DO NOT output the final answer to the original question.\n"
    "2. For EACH visual element you locate, output exactly ONE line in this strict format:\n"
    '   "short description of what this crop contains" -> [xmin, ymin, xmax, ymax]\n'
    "3. The description should briefly identify what this crop contains, matching your plan step "
    "(e.g., \"price of 青菜组合\", \"title of the chart\", \"the checkbox for option A\").\n"
    "4. Each bounding box must be normalized to the [0, 1000] range.\n\n"
    "--- EXAMPLE EXPECTED OUTPUT FORMAT ---\n"
    '"price of 青菜组合" -> [125, 340, 250, 380]\n'
    '"price of 豆制品拼盘" -> [510, 605, 750, 650]\n'
    '"total amount on the receipt" -> [800, 900, 950, 960]\n'
    "--------------------------------------\n\n"
    "Now, locate the visual elements for each plan step. Output ONLY the labeled coordinate lines "
    "in the format shown above — no extra reasoning, no final answer."
)

# 构造带历史的请求（与框架版本一致）
s2_messages = [
    {
        'role': 'user',
        'content': [build_text_content(prompt_s1), build_image_content(base64_image)]
    },
    {'role': 'assistant', 'content': plan_text},
    {'role': 'user', 'content': [build_text_content(prompt_s2)]}
]

s2_response = grounding_client.chat.completions.create(
    model=STAGE2_MODEL,
    temperature=0.0,
    max_tokens=6144,
    messages=s2_messages
)

grounding_text = s2_response.choices[0].message.content
_t2_elapsed = _time.time() - _t2_start
log(f"Stage 2 完成 ({_t2_elapsed:.0f}s)")
print(f"\n{'─' * 40}\n【坐标提取输出】({len(grounding_text)} chars):\n{grounding_text}\n{'─' * 40}\n")

# 保存 grounding 输出
with open(os.path.join(OUTPUT_DIR, "grounding_output.txt"), "w", encoding="utf-8") as f:
    f.write(grounding_text)


# ==========================================
# 6. 解析坐标与语义标签
# ==========================================

# 优先解析带语义标签的格式: "description" -> [xmin, ymin, xmax, ymax]
labeled_pattern = r'"([^"]*)"\s*->\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
labeled_matches = re.findall(labeled_pattern, grounding_text)

extracted_bboxes = []
crop_labels = []  # 与 extracted_bboxes 一一对应的语义标签

if labeled_matches:
    for m in labeled_matches:
        desc = m[0].strip()
        bbox = [int(m[1]), int(m[2]), int(m[3]), int(m[4])]
        extracted_bboxes.append(bbox)
        crop_labels.append(desc)
    log(f"从语义标签格式解析到 {len(extracted_bboxes)} 个标注坐标")
else:
    # 回退：解析无标签的裸坐标（兼容旧版输出）
    pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    matches = re.findall(pattern, grounding_text)
    seen = set()
    for m in matches:
        bbox = tuple(int(val) for val in m)
        if bbox not in seen:
            seen.add(bbox)
            extracted_bboxes.append(list(bbox))
            crop_labels.append(f"Region {len(crop_labels) + 1}")
    log(f"从裸坐标格式解析到 {len(extracted_bboxes)} 个坐标（无标签）")

# 裁剪图数量上限。超出部分不丢弃，而是以纯文本描述形式传给 Stage 3
cropped_bboxes = extracted_bboxes
cropped_labels = crop_labels
overflow_labels = []
if len(extracted_bboxes) > MAX_CROPS:
    log(f"bbox 数量 {len(extracted_bboxes)} 超过上限 {MAX_CROPS}，"
        f"前 {MAX_CROPS} 个裁剪为子图，剩余 {len(extracted_bboxes) - MAX_CROPS} 个仅保留文字描述")
    cropped_bboxes = extracted_bboxes[:MAX_CROPS]
    cropped_labels = crop_labels[:MAX_CROPS]
    overflow_labels = crop_labels[MAX_CROPS:]

log(f"裁剪子图: {len(cropped_bboxes)} 张 | 文字描述: {len(overflow_labels)} 个")


# ==========================================
# 7. Stage 3: 带标签的局部复审与最终推理
# ==========================================
if not cropped_bboxes:
    log("未提取到任何坐标，流水线终止。")
    sys.exit(1)

_t3_start = _time.time()
log("STAGE 3: 局部高清复审 (Labeled Crops + Original Thumbnail)")

# 构造 prompt_s3，根据是否有溢出区域动态生成说明
_overflow_note = ""
if overflow_labels:
    _overflow_items = "\n".join(f"  - {lbl}" for lbl in overflow_labels)
    _overflow_note = (
        f"\n\nIMPORTANT — CROP LIMIT REACHED:\n"
        f"Due to token limits, only {MAX_CROPS} out of {len(extracted_bboxes)} identified regions "
        f"could be provided as high-resolution crops above. "
        f"The following {len(overflow_labels)} region(s) are described below but NOT shown as crops — "
        f"you MUST locate them yourself in the ORIGINAL FULL IMAGE attached at the end:\n"
        f"{_overflow_items}\n"
        f"Use the original image to find and read the values for these missing regions. "
        f"They are equally important for answering the question correctly."
    )

prompt_s3 = (
    "Here are the high-resolution cropped images of the regions you just identified based on your plan. "
    "Each sub-image is preceded by a [Sub-image N: \"description\"] label telling you "
    "what the crop is supposed to contain — use these labels to understand which part of the "
    "problem each crop corresponds to."
    f"{_overflow_note}\n\n"
    "After these crops, a downscaled version of the ORIGINAL FULL IMAGE is also attached — "
    "use it for global context, spatial reference between regions, and to verify that your "
    "extracted data matches the overall layout.\n\n"
    "1. First, for each sub-image AND any missing region listed above, read the precise text "
    "or values visible. Use the label to understand what this value means in the context of "
    "the problem. Cross-reference with the original full image to confirm accuracy.\n"
    "2. Second, reflect on your initial plan and findings. Correct any hallucinations, "
    "missed items, or logical errors from your previous steps.\n"
    f"3. Finally, step-by-step, answer the original question: <q>{QUESTION}</q>. "
    "Enclose the final answer within the tags <a> and </a> like this: <a>your answer</a>."
)

# 构造 s3_user_content：文本提示 + 标签子图 + 溢出区域文字 + 缩略原图
s3_user_content = [build_text_content(prompt_s3)]

# 加载原图（用于裁剪和画框）
original_img = Image.open(LOCAL_IMAGE_PATH).convert('RGB')
clean_img = original_img.copy()
draw = ImageDraw.Draw(original_img)

for i, bbox in enumerate(cropped_bboxes):
    # 在子图前插入语义标签
    label = cropped_labels[i] if i < len(cropped_labels) else f"Region {i+1}"
    s3_user_content.append(build_text_content(f'[Sub-image {i+1}: "{label}"]'))

    xmin, ymin, xmax, ymax = bbox
    p_bbox = [
        int((xmin / 1000.0) * orig_w),
        int((ymin / 1000.0) * orig_h),
        int((xmax / 1000.0) * orig_w),
        int((ymax / 1000.0) * orig_h)
    ]

    # 裁剪并编码
    cropped_b64 = crop_and_encode(LOCAL_IMAGE_PATH, p_bbox, padding=PADDING,
                                  scale_factor=SCALE_FACTOR, max_edge=MAX_CROP_EDGE)
    s3_user_content.append(build_image_content(cropped_b64))

    # 在原图上画框（可视化）
    draw.rectangle(p_bbox, outline="#00FF00", width=4)
    # 标注序号
    draw.text((p_bbox[0], max(0, p_bbox[1] - 15)), f"#{i+1}", fill="#00FF00")

# 如果有溢出区域，以纯文本列表形式追加
if overflow_labels:
    _ov_list = "\n".join(f"  {j+1}. {lbl}" for j, lbl in enumerate(overflow_labels))
    s3_user_content.append(build_text_content(
        f"--- MISSING REGIONS (not cropped due to limit) ---\n"
        f"The following {len(overflow_labels)} region(s) were also identified "
        f"but could not be included as high-resolution crops. "
        f"Please locate them in the ORIGINAL FULL IMAGE below:\n"
        f"{_ov_list}"
    ))

# 附带缩略原图（最长边 1024px）
thumb_b64 = encode_pil_to_base64(clean_img, max_edge=MAX_ORIG_EDGE)
s3_user_content.append(build_image_content(thumb_b64))

log(f"裁剪完成: {len(cropped_bboxes)} 张子图 + {len(overflow_labels)} 个文字描述 + 1 张缩略原图")

# 构造 s3_history（重建对话历史，使用中性 prompt_s2 避免角色混淆）
prompt_s1_text = (
    f"You are an expert visual reasoning agent. We need to solve the following question: "
    f"<q>{QUESTION}</q>\n\n"
    "Create a detailed, atomic execution plan. Each step must be so specific that even a basic OCR model "
    "or entry-level grounding model can execute it without any reasoning.\n\n"
    "RULES for your plan:\n"
    "- Each step must target exactly ONE visual element\n"
    "- For each step, describe the element precisely with its semantic location\n"
    "- Include the expected data type for each step\n"
    "- Steps should be ordered logically: gather all raw data first, then compute\n\n"
    "OUTPUT FORMAT — a numbered list:\n"
    'Step N: [action] — locate "<exact element description>" (expected: <data type>)\n\n'
    "CRITICAL: Do NOT calculate any final answer. Do NOT output bounding box coordinates. "
    "Only output the atomic plan as a numbered list."
)

# 中性陈述替代原始 S2 指令，避免 Stage 3 模型角色混淆
prompt_s2_neutral = (
    "Based on your plan, the following visual regions were identified. "
    "Each line pairs a semantic label (describing what the crop contains) with its bounding box coordinates:"
)

s3_history = [
    {'role': 'user', 'content': [build_text_content(prompt_s1_text)]},
    {'role': 'assistant', 'content': plan_text},
    {'role': 'user', 'content': [build_text_content(prompt_s2_neutral)]},
    {'role': 'assistant', 'content': grounding_text},
    {'role': 'user', 'content': s3_user_content},
]

log("正在调用 Stage 3 模型进行推理...")
s3_response = stage3_client.chat.completions.create(
    model=STAGE3_MODEL,
    temperature=0.0,
    max_tokens=6144,
    messages=s3_history
)

final_answer = s3_response.choices[0].message.content
_t3_elapsed = _time.time() - _t3_start
_t_total = _time.time() - _t1_start

log(f"Stage 3 完成 ({_t3_elapsed:.0f}s)")
_crop_info = f'crops={len(cropped_bboxes)}'
if overflow_labels:
    _crop_info += f' overflow_text={len(overflow_labels)}'
log(f"总耗时: S1={_t1_elapsed:.0f}s | S2={_t2_elapsed:.0f}s | "
    f"S3={_t3_elapsed:.0f}s | 总计={_t_total:.0f}s | {_crop_info}")

print(f"\n{'=' * 60}")
print(f"【最终思维链与解答】:")
print(final_answer)
print(f"{'=' * 60}\n")


# ==========================================
# 8. 保存输出
# ==========================================

# 带框原图
annotated_path = os.path.join(OUTPUT_DIR, "annotated_image.jpg")
original_img.save(annotated_path, quality=95)
log(f"带标注框的原图已保存: {annotated_path}")

# 完整 pipeline 结果 JSON
result = {
    "image_path": LOCAL_IMAGE_PATH,
    "question": QUESTION,
    "original_size": [orig_w, orig_h],
    "plan_text": plan_text,
    "grounding_text": grounding_text,
    "n_total_bboxes": len(extracted_bboxes),
    "n_crops": len(cropped_bboxes),
    "n_overflow_text": len(overflow_labels),
    "crop_labels": cropped_labels,
    "overflow_labels": overflow_labels,
    "extracted_bboxes": extracted_bboxes,
    "final_answer": final_answer,
    "timing_s": {
        "stage1": round(_t1_elapsed, 1),
        "stage2": round(_t2_elapsed, 1),
        "stage3": round(_t3_elapsed, 1),
        "total": round(_t_total, 1),
    }
}
result_path = os.path.join(OUTPUT_DIR, "pipeline_result.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
log(f"完整结果已保存: {result_path}")

log("流水线执行完毕。")
