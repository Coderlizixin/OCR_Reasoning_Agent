# import pandas as pd
# import base64
# import io
# from PIL import Image
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# from openai import OpenAI
#
# # ==========================================
# # 1. 初始化大模型客户端 (全局复用)
# # ==========================================
# client = OpenAI(
#     api_key="sk-EMPTY",
#     base_url="http://10.3.112.34:22240/v1",
# )
#
#
# # ==========================================
# # 2. 内存中的图片处理函数
# # ==========================================
# def optimize_base64_image(base64_str, max_size=2048):
#     try:
#         img_data = base64.b64decode(base64_str)
#         with Image.open(io.BytesIO(img_data)) as img:
#             if img.width > max_size or img.height > max_size:
#                 img.thumbnail((max_size, max_size))
#
#             buffer = io.BytesIO()
#             if img.mode in ("RGBA", "P"):
#                 img = img.convert("RGB")
#             img.save(buffer, format="JPEG")
#
#             return base64.b64encode(buffer.getvalue()).decode('utf-8')
#     except Exception as e:
#         raise ValueError(f"图片处理失败: {e}")
#
#
# # ==========================================
# # 3. 单条数据请求任务
# # ==========================================
# def process_single_row(index, row):
#     item_id = row.get('index', index)
#     question = row.get('question', '')
#     raw_img_str = row.get('image', None)
#     ground_truth = row.get('answer', '')
#
#     if pd.isna(raw_img_str):
#         return {"id": item_id, "status": "error", "error_msg": "图片数据缺失"}
#
#     try:
#         optimized_base64 = optimize_base64_image(raw_img_str, max_size=2048)
#
#         prompt_text = (
#             "Answer the following question within the tags <q> and </q> step-by-step. "
#             "After presenting your reasoning, provide the final simple answer at last, "
#             "making sure to enclose it within the tags <a> and </a> like this: <a>your answer</a>. "
#             f"Question: <q>{question}</q>"
#         )
#
#         response = client.chat.completions.create(
#             model='Qwen3.5-9B',
#             temperature=0.0,
#             max_tokens=16384,
#             messages=[{ # type: ignore
#                 'role': 'user',
#                 'content': [
#                     {'type': 'text', 'text': prompt_text},
#                     {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{optimized_base64}"}}
#                 ],
#             }]
#         )
#
#         final_answer_text = response.choices[0].message.content
#         # 【新增】提取 finish_reason
#         finish_reason = response.choices[0].finish_reason
#
#         return {
#             "id": item_id,
#             "question": question,
#             "ground_truth": ground_truth,
#             "prediction": final_answer_text,
#             "finish_reason": finish_reason,  # 【新增】将结束原因存入字典
#             "status": "success"
#         }
#
#     except Exception as e:
#         return {"id": item_id, "status": "error", "error_msg": str(e)}
#
#
# # ==========================================
# # 4. 主程序：并行调度与统计
# # ==========================================
# if __name__ == "__main__":
#     tsv_path = r"C:\Users\19639\LMUData\OCR_Reasoning.tsv"
#
#     try:
#         print("正在加载数据集...")
#         df = pd.read_csv(tsv_path, sep='\t')
#         print(f"成功加载数据集，共 {len(df)} 条数据。")
#
#         results = []
#         MAX_WORKERS = 10
#
#         print(f"开始并行请求，最大并发数: {MAX_WORKERS}...")
#
#         with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#             futures = {
#                 executor.submit(process_single_row, idx, row): idx
#                 for idx, row in df.head(200).iterrows()
#             }
#
#             for future in tqdm(as_completed(futures), total=len(futures), desc="推理进度"):
#                 result = future.result()
#                 results.append(result)
#
#                 if result['status'] == 'error':
#                     tqdm.write(f"[Error] ID {result['id']} 失败: {result['error_msg']}")
#
#         print("\n所有任务处理完成！")
#
#         # 保存结果
#         results_df = pd.DataFrame(results)
#
#         # 【新增】统计 finish_reason
#         print("\n" + "=" * 40)
#         print("结束原因 (finish_reason) 统计:")
#         print("=" * 40)
#
#         # 只统计请求成功的条目
#         success_df = results_df[results_df['status'] == 'success']
#         if not success_df.empty and 'finish_reason' in success_df.columns:
#             # value_counts 会自动计算每种 reason 的数量
#             reason_stats = success_df['finish_reason'].value_counts()
#
#             for reason, count in reason_stats.items():
#                 print(f"[{reason}]: {count} 个")
#
#             # 顺便检查是否有因为 token 超限而截断的
#             if 'length' in reason_stats:
#                 print("\n注意：有模型输出因为达到 max_tokens 限制被截断，状态为 'length'。")
#         else:
#             print("没有成功返回的任务，无法统计 finish_reason。")
#         print("=" * 40 + "\n")
#
#         output_path = r"T:\PyCharmProject\OCR-Reasoning\OCR-Reasoning-main\outputs\Qwen3_5_maxtokens\OCR_Reasoning_Results200_16384.tsv"
#         results_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
#         print(f"结果已成功保存至: {output_path}")
#
#     except FileNotFoundError:
#         print(f"未找到文件: {tsv_path}")
#
#
# ########################################################################################################################

import pandas as pd
import base64
import io
import re
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# 1. 全局配置与客户端初始化
# ==========================================
client = OpenAI(
    api_key="sk-xxxxxx",
    base_url="http://10.3.112.34:22240/v1",
)


# ==========================================
# 2. 内存图像处理工具函数
# ==========================================
def process_base64_to_images(base64_str, max_size=2048):
    """
    将原始 Base64 解码为 PIL Image 对象，并生成一张用于全局视角的缩放版 Base64。
    """
    img_data = base64.b64decode(base64_str)
    original_img = Image.open(io.BytesIO(img_data))

    if original_img.mode in ("RGBA", "P"):
        original_img = original_img.convert("RGB")

    orig_w, orig_h = original_img.size

    global_img = original_img.copy()
    if global_img.width > max_size or global_img.height > max_size:
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
        xmin, xmax = xmax, xmin  # 左右颠倒则交换
    if ymin > ymax:
        ymin, ymax = ymax, ymin  # 上下颠倒则交换

    # 防止坐标一样导致长宽为0而报错
    if xmin == xmax:
        xmax += 1
    if ymin == ymax:
        ymax += 1

    # 边界保护
    crop_xmin = max(0, xmin - padding)
    crop_ymin = max(0, ymin - padding)
    crop_xmax = min(img.width, xmax + padding)
    crop_ymax = min(img.height, ymax + padding)

    cropped_img = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

    # 使用 Lanczos 算法进行高画质放大
    if scale_factor != 1.0:
        new_width = int(cropped_img.width * scale_factor)
        new_height = int(cropped_img.height * scale_factor)
        # 如果新长宽太小（比如大模型圈了个点），强行保护
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        magnified_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        magnified_img = cropped_img

    # 在内存中保存为高质量 JPEG
    buffer = io.BytesIO()
    magnified_img.save(buffer, format="JPEG", quality=95)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ==========================================
# 3. 单条数据的三阶段智能体处理逻辑
# ==========================================
def process_single_row(index, row):
    item_id = row.get('index', index)
    question = row.get('question', '')
    raw_img_str = row.get('image', None)
    ground_truth = row.get('answer', '')

    if pd.isna(raw_img_str):
        return {"id": item_id, "status": "error", "error_msg": "图片缺失"}

    try:
        # 准备全局图像数据
        original_img, (orig_w, orig_h), global_base64 = process_base64_to_images(raw_img_str)
        mime_type = "image/jpeg"

        # --------------------------------------------------
        # Stage 1: Planning (规划)
        # --------------------------------------------------
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
                {'type': 'image_url', 'image_url': {'url': f"data:{mime_type};base64,{global_base64}"}}
            ]
        }]

        res_plan = client.chat.completions.create(model='Qwen3.5-9B', temperature=0.0, max_tokens=6144, messages=conversation_history)
        plan_text = res_plan.choices[0].message.content
        conversation_history.append({'role': 'assistant', 'content': plan_text})

        # --------------------------------------------------
        # Stage 2: Grounding (视觉锚定提取坐标)
        # --------------------------------------------------
        conversation_history.append({
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': "Now, execute your plan. Scan the specific regions you identified and find the visual clues (text or values) required to solve the sub-problems. "
                        "For each clue, output its bounding box coordinates normalized to the [0, 1000] range in the format [xmin, ymin, xmax, ymax]. "
                        "Still, do NOT output the final answer yet."
            }]
        })

        res_grounding = client.chat.completions.create(model='Qwen3.5-9B', temperature=0.0, max_tokens=6144,
                                                       messages=conversation_history)
        grounding_text = res_grounding.choices[0].message.content
        conversation_history.append({'role': 'assistant', 'content': grounding_text})

        # 提取并去重坐标
        pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        matches = re.findall(pattern, grounding_text)
        extracted_bboxes = list({tuple(int(val) for val in match) for match in matches})

        # --------------------------------------------------
        # Stage 3: Reasoning (生成高清切图，复审并回答)
        # --------------------------------------------------
        if extracted_bboxes:
            round_3_content = [{
                'type': 'text',
                'text': "Here are the high-resolution cropped images of the regions you just identified based on your plan. "
                        "1. First, list the precise text or values visible in these cropped details.\n"
                        "2. Second, reflect on your initial plan and findings. Correct any hallucinations, missed items, or logical errors from your previous steps.\n"
                        f"3. Finally, step-by-step, answer the original question: <q>{question}</q>. Enclose the final answer within the tags <a> and </a> like this: <a>your answer</a>."
            }]

            for bbox in extracted_bboxes:
                xmin, ymin, xmax, ymax = bbox
                real_xmin = int((xmin / 1000.0) * orig_w)
                real_ymin = int((ymin / 1000.0) * orig_h)
                real_xmax = int((xmax / 1000.0) * orig_w)
                real_ymax = int((ymax / 1000.0) * orig_h)

                physical_bbox = [real_xmin, real_ymin, real_xmax, real_ymax]

                # 调用优化后的高清晰度放大裁切代码
                cropped_base64 = crop_and_encode(
                    img=original_img,
                    physical_bbox=physical_bbox,
                    padding=60,
                    scale_factor=3.0  # 3倍高清放大
                )

                round_3_content.append({
                    'type': 'image_url',
                    'image_url': {'url': f"data:image/jpeg;base64,{cropped_base64}"}
                })
        else:
            # 降级保护：没提取出坐标时直接回答
            round_3_content = [{
                'type': 'text',
                'text': "You didn't extract any specific visual coordinates. Please review your plan and the global image again, "
                        f"and step-by-step answer the original question: <q>{question}</q>. "
                        "Enclose the final answer within the tags <a> and </a> like this: <a>your answer</a>."
            }]

        conversation_history.append({'role': 'user', 'content': round_3_content})

        res_final = client.chat.completions.create(model='Qwen3.5-9B', temperature=0.0, max_tokens=6144,
                                                   messages=conversation_history)
        final_answer_text = res_final.choices[0].message.content
        finish_reason = res_final.choices[0].finish_reason

        return {
            "id": item_id,
            "question": question,
            "ground_truth": ground_truth,
            "plan": plan_text,
            "grounding": grounding_text,
            "extracted_boxes_count": len(extracted_bboxes),
            "prediction": final_answer_text,
            "finish_reason": finish_reason,
            "status": "success"
        }

    except Exception as e:
        return {"id": item_id, "status": "error", "error_msg": str(e)}


# ==========================================
# 4. 并行调度主程序
# ==========================================
if __name__ == "__main__":
    tsv_path = r"C:\Users\19639\LMUData\OCR_Reasoning.tsv"

    try:
        print("正在加载数据集...")
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"成功加载，共 {len(df)} 条数据。")

        results = []
        MAX_WORKERS = 6  # 图像放大会占用更多内存，适当降低并发数更稳定

        print(f"开始并行请求三阶段智能体(带LANCZOS放大)，最大并发数: {MAX_WORKERS}...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 测试用：只跑前 20 条。跑全量时删掉 .head(20)
            futures = {
                executor.submit(process_single_row, idx, row): idx
                for idx, row in df.head(200).iterrows()
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Agent 推理进度"):
                result = future.result()
                results.append(result)
                if result['status'] == 'error':
                    tqdm.write(f"[Error] ID {result['id']} 处理失败: {result['error_msg']}")

        print("\n所有任务处理完成！")
        results_df = pd.DataFrame(results)

        # ==========================================
        # finish_reason 统计
        # ==========================================
        print("\n" + "=" * 40)
        print("结束原因 (finish_reason) 统计:")
        print("=" * 40)

        # 只统计请求成功的条目
        success_df = results_df[results_df['status'] == 'success']
        if not success_df.empty and 'finish_reason' in success_df.columns:
            reason_stats = success_df['finish_reason'].value_counts()
            for reason, count in reason_stats.items():
                print(f"[{reason}]: {count} 个")

            if 'length' in reason_stats:
                print("\n注意：有模型输出因为达到 max_tokens 限制被截断，状态为 'length'。")
        else:
            print("没有成功返回的任务，无法统计 finish_reason。")
        print("=" * 40 + "\n")
        # ==========================================

        # 保存结果 (测试文件)
        output_path = r"T:\PyCharmProject\OCR-Reasoning\OCR-Reasoning-main\outputs\Qwen3_5_maxtokens\OCR_Reasoning_Results200_6144_Agent.tsv"
        results_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        print(f"结果已保存至: {output_path}")

    except FileNotFoundError:
        print(f"未找到文件: {tsv_path}")