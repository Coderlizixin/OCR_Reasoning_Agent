import os
import json
import pandas as pd
import base64
import io
import re
from PIL import Image, ImageDraw
from openai import OpenAI
from google import genai
from google.genai import types

# ==========================================
# 1. 全局配置与客户端初始化
# ==========================================
# huawei client
client = OpenAI(
    api_key="",
    base_url="http://10.3.112.34:22510/v1",
)

# model scope client
# client = OpenAI(
#     base_url='https://api-inference.modelscope.cn/v1',
#     api_key='ms-7ee485f4-550b-49f8-906a-216377ad5562',  # ModelScope Token
# )


# ==========================================
# 2. 内存图像处理工具函数
# ==========================================
def process_base64_to_images(base64_str, max_size=8192):
    """
    将原始 Base64 解码为 PIL Image，并生成 base64 编码。
    优先发送原图以最大化坐标精度；仅在图片巨大时缩放到 max_size 以控制 API 载荷。
    """
    img_data = base64.b64decode(base64_str)
    original_img = Image.open(io.BytesIO(img_data))

    if original_img.mode in ("RGBA", "P"):
        original_img = original_img.convert("RGB")

    orig_w, orig_h = original_img.size

    # 仅在图片极大时缩放，平时直接发送原图
    global_img = original_img.copy()
    if max(global_img.size) > max_size:
        global_img.thumbnail((max_size, max_size))
        print(f"  [图片缩放] {orig_w}x{orig_h} → {global_img.size[0]}x{global_img.size[1]}")

    buffer = io.BytesIO()
    global_img.save(buffer, format="JPEG", quality=95)
    global_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return original_img, (orig_w, orig_h), global_base64


# ==========================================
# 3. 边界框后处理：NMS 去重 + 尺寸过滤（适配细粒度）
# ==========================================
def bbox_iou(box1, box2):
    """计算两个 [xmin, ymin, xmax, ymax] 框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def filter_bboxes(bboxes, iou_threshold=0.3, min_size=8, max_size=980, max_boxes=20):
    """
    轻量 NMS 去重 + 尺寸过滤，保留细粒度框。

    Args:
        bboxes: [[xmin, ymin, xmax, ymax], ...] 归一化 [0,1000]
        iou_threshold: IoU 超过此阈值视为重复（降低以保留更多框）
        min_size: 框的最小宽/高（降低以保留小元素如文字标签）
        max_size: 框的最大宽/高（仅过滤覆盖全图的异常大框）
        max_boxes: 最多保留的框数（提高以适应细粒度）

    Returns:
        过滤后的 bbox 列表
    """
    if not bboxes:
        return []

    # 1. 过滤尺寸异常的框
    valid = []
    for b in bboxes:
        w, h = b[2] - b[0], b[3] - b[1]
        if w <= 0 or h <= 0:
            continue
        if w < min_size or h < min_size:
            continue  # 太小的噪点
        if w > max_size or h > max_size:
            continue  # 整张图的异常大框
        if all(0 <= v <= 1000 for v in b) and b[0] < b[2] and b[1] < b[3]:
            valid.append(b)

    if not valid:
        return []

    # 2. 按框面积升序排列（小框优先保留，因为细粒度元素较小）
    valid.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    # 3. NMS 去重（仅移除高度重叠的框）
    kept = []
    for b in valid:
        is_dup = False
        for k in kept:
            if bbox_iou(b, k) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(b)
        if len(kept) >= max_boxes:
            break

    return kept


# ==========================================
# 4. 边界框绘制函数
# ==========================================
def draw_bboxes_on_image(image, bboxes, orig_w, orig_h, output_path):
    """
    在图片上绘制细粒度边界框并保存。
    每个框用不同颜色标注，并带有序号。
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    colors = [
        "#00FF00", "#FF0000", "#0000FF", "#FFA500", "#FF00FF",
        "#00FFFF", "#FFFF00", "#800080", "#008000", "#FF69B4",
        "#FFD700", "#00CED1", "#DC143C", "#7FFF00", "#FF1493",
        "#1E90FF", "#FF6347", "#32CD32", "#8A2BE2", "#FF8C00"
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
            width=4
        )
        draw.text(
            (real_xmin + 4, real_ymin + 4),
            str(i + 1),
            fill=color
        )

    annotated.save(output_path, quality=95)
    return annotated


# ==========================================
# 5. 单条数据处理函数
# ==========================================
def process_single_row(index, row, output_dir):
    """
    将单条 OCR-Reasoning 数据发送给 Qwen3.5-397B-A17B，
    让模型输出细粒度的视觉元素 bounding box，并绘制。
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

        # 截断过长的推理文本
        # if len(reasoning) > 2000:
        #     reasoning_short = reasoning[:1500] + "\n...(中间省略)...\n" + reasoning[-500:]
        # else:
        #     reasoning_short = reasoning
        # reasoning_short = reasoning

        # ============================================================
        # 细粒度视觉元素定位提示词
        # ============================================================
        # prompt = f"""You are a visual grounding specialist. Your task is to generate precise bounding boxes for the visual evidence required to solve complex text-rich reasoning problems.
        #
        # The images come from diverse scenarios including charts, financial documents, math/physics problems, receipts, schedules, and logical maps.
        #
        # === QUESTION ===
        #         [QUESTION]: {question}
        #
        # === ANSWER & REASONING ===
        #         [CORRECT ANSWER]: {ground_truth}
        #         [REASONING PROCESS]: {reasoning_short}
        #
        # === INSTRUCTIONS ===
        #         Analyze the [REASONING PROCESS] step-by-step. Identify SPECIFIC visual element in the image that the reasoning explicitly relies on. Only annotate what the reasoning ACTUALLY uses.
        #
        #         Element Types to capture based on the scenario:
        #         - Documents/Receipts/Menus: Specific prices, item names, dates, quantities, etc.
        #         - Charts/Graphs: Specific bars, data lines, legend labels, specific axis values, etc.
        #         - Math/Science: Handwritten formulas, geometric shapes/angles, circuit diagrams, etc.
        #         - Tables/Schedules: Specific cell values, row/column headers, etc.
        #         - Logic/Spatial: Specific map regions, route nodes, layout blocks, etc.
        #
        #         === GROUNDING RULES ===
        #         1. STRICT RELEVANCE: ONLY box elements explicitly mentioned or logically required by the [REASONING PROCESS]. Do not box unrelated background data.
        #         2. TIGHT & ATOMIC: Box the smallest meaningful unit. The box should enclose ONLY that element — not its neighbors, not its container, not the surrounding whitespace. Do NOT box an entire table or chart. Box the specific text phrase, number, or data point.
        #         3. EXHAUSTIVE: If the reasoning sums 4 numbers, you must provide 4 separate bounding boxes for those 4 numbers.
        #         4. ALIGNED: Boxes must be aligned with the elements in original image.
        #
        #         === COORDINATE FORMAT ===
        #         Normalized to [0, 1000]: [xmin, ymin, xmax, ymax] where (0,0) is top-left and (1000,1000) is bottom-right. All values must be integers.
        #
        #         === SELF-CHECK (do this silently before outputting) ===
        #         1. Does the reasoning actually reference this element? If not, skip it.
        #         2. Is the box tight around the element (not loose)?
        #         3. Are the coordinates in the correct [0,1000] range?
        #         4. Is this box distinct from other boxes (not heavily overlapping)?
        #         5. Is this box align with the element in original image?
        #
        #         === OUTPUT FORMAT ===
        #         Output ONLY a valid JSON array of objects. Each object: {{"desc": "<what this element is>", "bbox": [xmin, ymin, xmax, ymax]}}
        #         Each object must have "desc" (a brief description of the element) and "bbox" (the coordinate array).
        #         Example Output:
        #         [
        #           {{"desc": "Price of Item A ($15.99)", "bbox": [120, 300, 250, 340]}},
        #           {{"desc": "Orange legend label 'Drama'", "bbox": [800, 50, 900, 80]}},
        #           {{"desc": "Handwritten angle alpha in the triangle", "bbox": [400, 500, 450, 550]}},
        #           {{"desc": "Value 42 in the table cell", "bbox": [600, 700, 650, 730]}}
        #         ]
        #
        #         Now, generate the JSON array for the current image:"""

        prompt = f"""You are a visual grounding specialist. Your task is to generate precise bounding boxes for the visual evidence required to solve complex text-rich reasoning problems.

The images come from diverse scenarios including charts, financial documents, math/physics problems, receipts, schedules, and logical maps.

=== QUESTION ===
        [QUESTION]: {question}
=== ANSWER & REASONING ===
        [CORRECT ANSWER]: {ground_truth}
        [REASONING PROCESS]: {reasoning}

=== INSTRUCTIONS ===
        Based on the question and the image, you need to arrive at the answer through a series of step-by-step reasoning. Please identify the necessary elements that must be used in the reasoning process.
        
        Element Types to capture based on the scenario:
        - Documents/Receipts/Menus: Specific prices, item names, dates, quantities, etc.
        - Charts/Graphs: Specific bars, data lines, related legend labels, specific axis values, etc.
        - Math/Science: Handwritten formulas, geometric shapes/angles, circuit diagrams, etc.
        - Tables/Schedules: Specific cell values, row/column headers, etc.
        - Logic/Spatial: Specific map regions, route nodes, layout blocks, etc.

        === GROUNDING RULES ===
        1. STRICT RELEVANCE: ONLY box elements explicitly mentioned or logically required by the [REASONING PROCESS]. Do not box unrelated background data.
        2. TIGHT & ATOMIC: Box the smallest meaningful unit. The box should enclose ONLY that element — not its neighbors, not its container, not the surrounding whitespace. Do NOT box an entire table or chart. Box the specific text phrase, number, or data point.
        3. EXHAUSTIVE: If the reasoning sums 4 numbers, you must provide 4 separate bounding boxes for those 4 numbers.
        4. ALIGNED: Boxes must be aligned with the elements in original image.

        === COORDINATE FORMAT ===
        Normalized to [0, 1000]: [xmin, ymin, xmax, ymax] where (0,0) is top-left and (1000,1000) is bottom-right. All values must be integers.

        === SELF-CHECK (do this silently before outputting) ===
        1. Does the reasoning actually reference this element? If not, skip it.
        2. Is the box tight around the element (not loose)?
        3. Are the coordinates in the correct [0,1000] range?
        4. Is this box distinct from other boxes (not heavily overlapping)?
        5. Is this box align with the element in original image?

        === OUTPUT FORMAT ===
        Output ONLY a valid JSON array of objects. Each object: {{"desc": "<what this element is>", "bbox": [xmin, ymin, xmax, ymax]}}
        Each object must have "desc" (a brief description of the element) and "bbox" (the coordinate array).
        Example Output:
        [
          {{"desc": "Price of Item A ($15.99)", "bbox": [120, 300, 250, 340]}},
          {{"desc": "Orange legend label 'Drama'", "bbox": [800, 50, 900, 80]}},
          {{"desc": "Handwritten angle alpha in the triangle", "bbox": [400, 500, 450, 550]}},
          {{"desc": "Value 42 in the table cell", "bbox": [600, 700, 650, 730]}}
        ]

        Now, generate the JSON array for the current image:"""


        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{global_base64}"},
                 # 'min_pixels': 2500 * 32 * 32,
                 # 'max_pixels': 4900 * 32 * 32,
                 }
            ]
        }]

        response = client.chat.completions.create(
            model='Qwen3.5-122B-A10B',
            temperature=0.0,
            max_tokens=8192,
            messages=messages,
            extra_body={
                "mm_processor_kwargs": {
                    "images_kwargs": {
                        "size": {
                            "longest_edge": 8192 * 32 * 32,
                            "shortest_edge": 4096 * 32 * 32,
                        }
                    }
                }
            },
            stream=True
        )

        # response_text = response.choices[0].message.content
        response_text = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
        print("===response_text===")
        print(f"  [模型原始输出]: {response_text}")

        # ============================================================
        # 解析 bounding box 坐标（同时保存描述文本）
        # ============================================================
        extracted_items = []  # [{"desc": str, "bbox": [xmin,ymin,xmax,ymax]}, ...]

        # 路径1: JSON 解析（支持尾随文本）
        # 使用 raw_decode 逐步解析，处理模型在 JSON 后追加解释文本的情况
        json_parsed = False
        array_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
        if not array_match:
            # 尝试跨行匹配（允许嵌套的 bbox 数组）
            array_match = re.search(r'\[.*?\]', response_text, re.DOTALL)

        if array_match:
            try:
                decoder = json.JSONDecoder()
                # raw_decode 返回 (obj, end_index)，可处理尾随文本
                parsed, _ = decoder.raw_decode(array_match.group(0))
                for item in parsed:
                    if isinstance(item, dict) and 'bbox' in item:
                        bbox = item['bbox']
                        if len(bbox) == 4:
                            extracted_items.append({
                                "desc": str(item.get('desc', '')),
                                "bbox": [int(v) for v in bbox]
                            })
                if extracted_items:
                    print(f"  [JSON解析成功]: {len(extracted_items)} 个框")
                    json_parsed = True
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                print(f"  [JSON解析失败，回退正则]: {e}")

        # 路径2: 正则回退 — 同时提取坐标和描述
        if not extracted_items:
            # 先尝试逐对象匹配 {"desc": "...", "bbox": [...]}
            obj_pattern = r'\{"desc"\s*:\s*"([^"]*?)"\s*,\s*"bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\}'
            for m in re.finditer(obj_pattern, response_text):
                extracted_items.append({
                    "desc": m.group(1),
                    "bbox": [int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))]
                })
            if extracted_items:
                print(f"  [正则逐对象提取]: {len(extracted_items)} 个框（含 desc）")
            else:
                # 纯坐标回退
                pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
                matches = re.findall(pattern, response_text)
                seen = set()
                for match in matches:
                    bbox_tuple = tuple(int(val) for val in match)
                    if bbox_tuple not in seen:
                        seen.add(bbox_tuple)
                        extracted_items.append({
                            "desc": "",
                            "bbox": list(bbox_tuple)
                        })
                print(f"  [纯坐标提取]: {len(extracted_items)} 个框（无 desc）")

        # 分离出纯坐标列表用于 NMS
        extracted_bboxes = [item['bbox'] for item in extracted_items]

        # NMS + 尺寸过滤（细粒度模式）
        filtered_bboxes = filter_bboxes(
            extracted_bboxes,
            iou_threshold=0.3,
            min_size=8,
            max_size=980,
            max_boxes=20
        )

        # 构建过滤后的详情列表（bbox → 物理坐标 + 描述）
        filtered_items = []
        for bbox in filtered_bboxes:
            # 尝试匹配原始描述
            desc = ""
            for item in extracted_items:
                if item['bbox'] == bbox:
                    desc = item['desc']
                    break
            phys_bbox = [
                int((bbox[0] / 1000.0) * orig_w),
                int((bbox[1] / 1000.0) * orig_h),
                int((bbox[2] / 1000.0) * orig_w),
                int((bbox[3] / 1000.0) * orig_h)
            ]
            filtered_items.append({
                "desc": desc,
                "bbox_norm": bbox,
                "bbox_phys": phys_bbox
            })

        print(f"  原始 {len(extracted_bboxes)} 框 → 过滤后 {len(filtered_bboxes)} 框")

        safe_id = str(item_id).replace('/', '_').replace('\\', '_')
        stage1_items = filtered_items  # 保存第一阶段结果

        # ================================================================
        # 绘制第一阶段标注图（审查前）
        # ================================================================
        output_image_initial_path = None
        if stage1_items:
            initial_filename = f"annotated_{safe_id}_stage1_initial.jpg"
            output_image_initial_path = os.path.join(output_dir, initial_filename)
            stage1_bboxes = [item['bbox_norm'] for item in stage1_items]
            draw_bboxes_on_image(
                image=original_img,
                bboxes=stage1_bboxes,
                orig_w=orig_w,
                orig_h=orig_h,
                output_path=output_image_initial_path
            )
            print(f"  第一阶段标注图已保存: {output_image_initial_path}")

        # ================================================================
        # 二阶段 ReAct 精修（暂时禁用）
        # ================================================================
        # refine_text = ""
        # if stage1_items:
        #     print("  [ReAct精修阶段] 逐个裁剪放大，在放大图中精确定位...")
        #     refined_items, refine_text = refine_bboxes(
        #         original_img=original_img,
        #         orig_w=orig_w,
        #         orig_h=orig_h,
        #         global_base64=global_base64,
        #         initial_items=stage1_items,
        #         question=question,
        #         ground_truth=ground_truth,
        #         reasoning=reasoning
        #     )
        #     stage2_items = refined_items if refined_items else stage1_items
        # else:
        #     stage2_items = stage1_items
        refine_text = ""
        stage2_items = stage1_items
        output_image_path = output_image_initial_path
        json_output_path = None

        return {
            "id": item_id,
            "question": question,
            "ground_truth": ground_truth,
            "reasoning": reasoning,
            "model_response_stage1": response_text,
            "model_response_stage2": "",
            "raw_bbox_count": len(extracted_bboxes),
            "initial_bbox_count": len(stage1_items),
            "refined_bbox_count": len(stage1_items),
            "extracted_bboxes": stage1_bboxes if stage1_items else [],
            "bbox_details_stage1": stage1_items,
            "bbox_details_stage2": stage1_items,
            "annotated_image_initial": output_image_initial_path,
            "annotated_image_refined": output_image_initial_path,
            "bbox_json": None,
            "status": "success"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"id": item_id, "status": "error", "error_msg": str(e)}


# ==========================================
# 6. ReAct 二阶段精修：裁剪放大 → 模型在放大图中精确定位
# ==========================================
def _encode_img_to_base64(img):
    """PIL Image → base64 JPEG 字符串"""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def react_refine_one_bbox(original_img, orig_w, orig_h, bbox_norm, desc, expand_ratio=1.25):
    """
    ReAct 单框精修:
    1. 将 Stage 1 bbox 扩展 expand_ratio 倍
    2. 裁剪放大后送模型精确定位
    3. 将 crop-relative 坐标映射回全图

    返回: refined bbox_norm [xmin, ymin, xmax, ymax] in [0,1000]
    """
    xmin_n, ymin_n, xmax_n, ymax_n = bbox_norm

    if not desc or not desc.strip():
        print(f"      [ReAct] desc 为空，跳过精修，保留 Stage 1 坐标")
        return bbox_norm

    bbox_w = xmax_n - xmin_n
    bbox_h = ymax_n - ymin_n

    # ---- Step 1: 扩展 bbox -------------------------------------------------
    expand_w = int(bbox_w * (expand_ratio - 1) / 2)
    expand_h = int(bbox_h * (expand_ratio - 1) / 2)

    crop_xmin = max(0, xmin_n - expand_w)
    crop_ymin = max(0, ymin_n - expand_h)
    crop_xmax = min(1000, xmax_n + expand_w)
    crop_ymax = min(1000, ymax_n + expand_h)

    # ---- Step 2: 裁剪 -------------------------------------------------------
    px_crop = [
        int(crop_xmin / 1000.0 * orig_w),
        int(crop_ymin / 1000.0 * orig_h),
        int(crop_xmax / 1000.0 * orig_w),
        int(crop_ymax / 1000.0 * orig_h),
    ]
    cropped = original_img.crop(px_crop)
    crop_w, crop_h = cropped.size

    # ---- Step 3: 放大（长边到 640px）-----------------------------------------
    TARGET_SIZE = 640
    scale = TARGET_SIZE / max(crop_w, crop_h)
    if scale > 1:
        cropped = cropped.resize(
            (int(crop_w * scale), int(crop_h * scale)), Image.LANCZOS)
        crop_w, crop_h = cropped.size

    print(f"      裁剪: {px_crop[2]-px_crop[0]}×{px_crop[3]-px_crop[1]}px → 放大后: {crop_w}×{crop_h}px")

    crop_base64 = _encode_img_to_base64(cropped)

    # ---- Step 4: 模型精确定位 ------------------------------------------------
    refine_prompt = f"""This is a zoomed-in crop containing: "{desc}"

Output the bounding box of this element. Format: [xmin, ymin, xmax, ymax]
Coordinates: [0,1000] relative to this crop, (0,0)=top-left.

Example: [120, 400, 180, 460]

Reply with ONLY the four numbers. No words."""

    try:
        crop_response = client.chat.completions.create(
            model='Qwen3.5-9B',
            temperature=0.0,
            max_tokens=256,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': refine_prompt},
                    {'type': 'image_url',
                     'image_url': {'url': f"data:image/jpeg;base64,{crop_base64}"}},
                ]
            }],
            extra_body={
                "mm_processor_kwargs": {
                    "images_kwargs": {
                        "size": {
                            "longest_edge": 8192 * 32 * 32,
                            "shortest_edge": 4096 * 32 * 32,
                        }
                    }
                }
            },
            stream=True
        )

        crop_text = ""
        for chunk in crop_response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                crop_text += chunk.choices[0].delta.content

        print(f"      [模型输出]: {crop_text[:200]}")

        # 解析坐标
        match = re.search(
            r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]',
            crop_text)

        if not match:
            # JSON 回退
            match = re.search(
                r'"bbox"\s*:\s*\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]',
                crop_text)

        if not match:
            print(f"      [ReAct] 未解析到坐标，保留 Stage 1")
            return bbox_norm

        cxmin, cymin, cxmax, cymax = [float(v) for v in match.groups()]

        # ---- Step 5: 映射回全图坐标 -----------------------------------------
        crop_norm_w = crop_xmax - crop_xmin
        crop_norm_h = crop_ymax - crop_ymin

        refined = [
            max(0, min(1000, int(round(crop_xmin + cxmin / 1000.0 * crop_norm_w)))),
            max(0, min(1000, int(round(crop_ymin + cymin / 1000.0 * crop_norm_h)))),
            max(0, min(1000, int(round(crop_xmin + cxmax / 1000.0 * crop_norm_w)))),
            max(0, min(1000, int(round(crop_ymin + cymax / 1000.0 * crop_norm_h)))),
        ]

        # 安全阀：偏移不超过原始 bbox 宽高的 2 倍
        shift_tolerance = max(bbox_w * 2, 80), max(bbox_h * 2, 80)
        dx = abs(refined[0] - xmin_n)
        dy = abs(refined[1] - ymin_n)
        if dx > shift_tolerance[0] or dy > shift_tolerance[1]:
            print(f"      [ReAct] 偏移过大 (dx={dx:.0f}, dy={dy:.0f})，保留 Stage 1")
            return bbox_norm

        print(f"      [ReAct] 精修成功: [{xmin_n},{ymin_n},{xmax_n},{ymax_n}] → {refined}")
        return refined

    except Exception as e:
        print(f"      [ReAct异常: {e}] 保留 Stage 1")
        return bbox_norm


def refine_bboxes(original_img, orig_w, orig_h, global_base64,
                  initial_items, question, ground_truth, reasoning):
    """
    ReAct 二阶段精修：
    对每个 Stage 1 bbox，裁剪放大目标区域 → 模型在放大图中精确标注 → 映射回全图。

    核心优势：模型在裁剪放大图中做定位远比在全图上做坐标预测准确。
    """
    if not initial_items:
        print("  [ReAct精修跳过] 没有 bbox")
        return initial_items, ""

    print(f"  [ReAct精修] 对 {len(initial_items)} 个 bbox 逐个裁剪放大精修...")

    refined_items = []
    refine_logs = []

    for i, item in enumerate(initial_items):
        desc = item['desc']
        bbox_norm = item['bbox_norm']

        print(f"    [{i + 1}/{len(initial_items)}] 精修: \"{desc[:60]}...\"" if len(desc) > 60
              else f"    [{i + 1}/{len(initial_items)}] 精修: \"{desc}\"")

        refined_bbox_norm = react_refine_one_bbox(
            original_img, orig_w, orig_h, bbox_norm, desc, expand_ratio=1.25
        )

        phys_bbox = [
            int((refined_bbox_norm[0] / 1000.0) * orig_w),
            int((refined_bbox_norm[1] / 1000.0) * orig_h),
            int((refined_bbox_norm[2] / 1000.0) * orig_w),
            int((refined_bbox_norm[3] / 1000.0) * orig_h)
        ]

        refined_items.append({
            "desc": desc,
            "bbox_norm": refined_bbox_norm,
            "bbox_phys": phys_bbox
        })

        # 记录精修变化
        dx = [refined_bbox_norm[j] - bbox_norm[j] for j in range(4)]
        refine_logs.append(f"    [{i}]: Δ=({dx[0]:+d},{dx[1]:+d},{dx[2]:+d},{dx[3]:+d})")

    # 汇总日志
    log_text = "\n".join(refine_logs) if refine_logs else ""
    if log_text:
        print(f"  [ReAct精修变化]:\n{log_text}")

    return refined_items, log_text


# ==========================================
# 7. 主程序
# ==========================================
if __name__ == "__main__":
    tsv_path = r"C:\Users\19639\LMUData\OCR_Reasoning.tsv"
    output_dir = r"test_case/bbox_annotations"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载随机索引列表
        indices_path = r"test_case/random_indices.json"
        with open(indices_path, 'r', encoding='utf-8') as f:
            target_indices = json.load(f)
        print(f"目标索引: {len(target_indices)} 个")

        print("正在加载数据集...")
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"成功加载，共 {len(df)} 条数据。\n")

        # 按 index 列筛选目标实例
        subset = df[df['index'].isin(target_indices)]
        print(f"匹配到 {len(subset)} 条目标实例")

        results = []
        for i, (idx, row) in enumerate(subset.iterrows()):
            item_id = row.get('index', idx)
            print(f"\n{'=' * 60}")
            print(f"[{i+1}/{len(subset)}] 处理 ID={item_id}")
            q = str(row.get('question', ''))
            print(f"问题: {q[:150]}{'...' if len(q) > 150 else ''}")
            print(f"{'=' * 60}")

            result = process_single_row(idx, row, output_dir)
            results.append(result)

            if result['status'] == 'success':
                print(f"结果: {result['initial_bbox_count']} 个框")
                if result.get('annotated_image_initial'):
                    print(f"标注图: {result['annotated_image_initial']}")
            else:
                print(f"错误: {result['error_msg']}")

        results_df = pd.DataFrame(results)
        results_output = os.path.join(output_dir, "bbox_results.tsv")
        results_df.to_csv(results_output, sep='\t', index=False, encoding='utf-8')
        print(f"\n{'=' * 60}")
        print(f"完成 {len(results)}/{len(subset)} 条")
        print(f"汇总结果: {results_output}")
        print(f"标注图片目录: {output_dir}")

    except FileNotFoundError:
        print(f"未找到文件: {tsv_path}")
