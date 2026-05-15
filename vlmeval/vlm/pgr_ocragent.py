import os
import io
import re
import base64
import json
import tempfile
from PIL import Image
from vlmeval.api.gpt import OpenAIWrapper


class ThreeStageOCRAgent(OpenAIWrapper):
    def __init__(self, padding=60, scale_factor=3.0, **kwargs):
        """
        初始化三阶段智能体 (纯内存操作优化版)。
        padding: 裁剪时的边缘扩充像素。
        scale_factor: 局部图的放大倍数。
        """
        super().__init__(**kwargs)
        self.padding = padding
        self.scale_factor = scale_factor
        # 确保输出长度足够容纳长思维链
        if 'max_tokens' not in kwargs:
            self.max_tokens = 4096

    def _crop_and_encode_base64(self, image_path, physical_bbox):
        """【核心优化】执行物理裁剪、放大，并直接在内存中返回 Base64，绝部落盘"""
        with Image.open(image_path).convert('RGB') as img:
            xmin, ymin, xmax, ymax = physical_bbox
            # 计算带 padding 的裁剪区域
            c_xmin = max(0, xmin - self.padding)
            c_ymin = max(0, ymin - self.padding)
            c_xmax = min(img.width, xmax + self.padding)
            c_ymax = min(img.height, ymax + self.padding)

            cropped = img.crop((c_xmin, c_ymin, c_xmax, c_ymax))

            if self.scale_factor != 1.0:
                new_size = (int(cropped.width * self.scale_factor),
                            int(cropped.height * self.scale_factor))
                cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)

            # 纯内存转换，不产生任何临时文件
            buffer = io.BytesIO()
            cropped.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def prepare_itlist(self, inputs):
        """【核心适配】扩展父类的解析逻辑，支持直接传递 Base64 字符串"""
        content_list = []
        for msg in inputs:
            if msg['type'] == 'image_base64':
                # 拦截内存图，直接构造 OpenAI API 要求的格式
                img_struct = dict(url=f"data:image/jpeg;base64,{msg['value']}", detail=self.img_detail)
                content_list.append(dict(type='image_url', image_url=img_struct))
            else:
                # 普通文本或本地原图路径，原封不动地交给父类去处理
                content_list.extend(super().prepare_itlist([msg]))
        return content_list

    def generate_inner(self, inputs, **kwargs):
        """
        核心推理逻辑，通过 super().generate_inner 维持上下文。
        """
        # 1. 提取原始图片路径和问题
        image_path = None
        original_question = ""
        for item in inputs:
            if item['type'] == 'image':
                image_path = item['value']
            elif item['type'] == 'text':
                original_question += item['value']

        # 如果输入不含图片，直接退化为普通 API 调用
        if not image_path:
            return super().generate_inner(inputs, **kwargs)

        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        # ------------------------------------------
        # STAGE 1: Planning (宏观规划)
        # ------------------------------------------
        prompt_s1 = (
            f"You are an expert visual reasoning agent. We need to solve the following question: <q>{original_question}</q>\n"
            "Before finding specific clues, please make a detailed step-by-step plan to break down this problem into progressive sub-problems. "
            "Analyze which general areas or categories in the image you need to focus on. "
            "CRITICAL: Do NOT calculate the final answer, and do NOT output any bounding box coordinates yet. Just output your plan."
        )

        s1_inputs = [
            {'type': 'text', 'value': prompt_s1},
            {'type': 'image', 'value': image_path}  # 原图路径交给父类读取
        ]

        ret, plan_text, resp = super().generate_inner(s1_inputs, **kwargs)
        if ret != 0: return ret, plan_text, resp

        # ------------------------------------------
        # STAGE 2: Grounding (视觉锚定)
        # ------------------------------------------
        prompt_s2 = (
            "Now, execute your plan. Scan the specific regions you identified and find the visual clues (text or values) required to solve the sub-problems. "
            "For each clue, output its bounding box coordinates normalized to the [0, 1000] range in the format [xmin, ymin, xmax, ymax]. "
            "Still, do NOT output the final answer yet."
        )

        history = [
            {'role': 'user', 'content': s1_inputs},
            {'role': 'assistant', 'content': [{'type': 'text', 'value': plan_text}]},
            {'role': 'user', 'content': [{'type': 'text', 'value': prompt_s2}]}
        ]

        ret, grounding_text, resp = super().generate_inner(history, **kwargs)
        if ret != 0: return ret, grounding_text, resp

        # ------------------------------------------
        # STAGE 3: Review & Reasoning (微观推理)
        # ------------------------------------------
        pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        matches = re.findall(pattern, grounding_text)

        extracted_bboxes = []
        seen = set()
        for m in matches:
            bbox = tuple(int(val) for val in m)
            if bbox not in seen:
                seen.add(bbox)
                extracted_bboxes.append(list(bbox))

        prompt_s3 = (
            "Here are the high-resolution cropped images of the regions you just identified based on your plan. "
            "1. First, list the precise text or values visible in these cropped details.\n"
            "2. Second, reflect on your initial plan and findings. Correct any hallucinations, missed items, or logical errors from your previous steps.\n"
            f"3. Finally, step-by-step, answer the original question: <q>{original_question}</q>. Enclose the final answer within the tags <a> and </a> like this: <a>your answer</a>."
        )

        s3_user_content = [{'type': 'text', 'value': prompt_s3}]

        if extracted_bboxes:
            for bbox in extracted_bboxes:
                xmin, ymin, xmax, ymax = bbox
                p_bbox = [
                    int((xmin / 1000.0) * orig_w),
                    int((ymin / 1000.0) * orig_h),
                    int((xmax / 1000.0) * orig_w),
                    int((ymax / 1000.0) * orig_h)
                ]
                # 获取内存中的 Base64 编码
                b64_string = self._crop_and_encode_base64(image_path, p_bbox)
                # 使用自定义类型 'image_base64'
                s3_user_content.append({'type': 'image_base64', 'value': b64_string})
        else:
            s3_user_content.append({'type': 'text',
                                    'value': "Note: No specific coordinates were extracted. Please reason based on the original image."})
            s3_user_content.append({'type': 'image', 'value': image_path})

        history.append({'role': 'assistant', 'content': [{'type': 'text', 'value': grounding_text}]})
        history.append({'role': 'user', 'content': s3_user_content})

        # 最终调用父类，父类会通过我们重写的 prepare_itlist 无缝解析内存 Base64
        return super().generate_inner(history, **kwargs)