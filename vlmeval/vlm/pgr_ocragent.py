import os
import io
import re
import base64
import json
import tempfile
from PIL import Image
from vlmeval.api.gpt import OpenAIWrapper, _log_diag


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
            self.max_tokens = 6144
        # Stage 1: 独立 9B 实例 (端口 22250)，避免与 Stage 3 抢占同一服务端
        self.stage1_agent = OpenAIWrapper(
            model='Qwen3.5-9B',
            api_base='http://10.3.112.34:22630/v1/chat/completions',
            # model='Qwen3.5-122B-A10B',
            # api_base='http://10.3.112.34:22510/v1/chat/completions',
            key='sk-EMPTY',
            temperature=0.0,
            max_tokens=6144,
            img_size=self.img_size,
            img_detail=self.img_detail,
            extra_body=self.extra_body,
            verbose=self.verbose
        )
        # Stage 2: 122B 强模型做 grounding (端口 22510)
        self.grounding_agent = OpenAIWrapper(
            model='Qwen3.5-9B',
            api_base='http://10.3.112.34:22630/v1/chat/completions',
            # model='Qwen3.5-122B-A10B',
            # api_base='http://10.3.112.34:22510/v1/chat/completions',
            key='ms-EMPTY',
            temperature=0.0,
            max_tokens=6144,
            img_size=self.img_size,
            img_detail=self.img_detail,
            extra_body=self.extra_body,
            verbose=self.verbose
        )

    def _crop_and_encode_base64(self, image_path, physical_bbox, max_edge=None):
        """【核心优化】执行物理裁剪、放大，并直接在内存中返回 Base64，绝部落盘。
        max_edge: 裁剪图最长边上限（像素），用于控制 visual token 数，None 则不限制。"""
        with Image.open(image_path).convert('RGB') as img:
            xmin, ymin, xmax, ymax = physical_bbox
            # 防止坐标错误
            if xmin > xmax:
                xmin, xmax = xmax, xmin  # 左右颠倒则交换
            if ymin > ymax:
                ymin, ymax = ymax, ymin  # 上下颠倒则交换

            # 防止坐标一样导致长宽为0而报错
            if xmin == xmax:
                xmax += 1
            if ymin == ymax:
                ymax += 1

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

            # 限制裁剪图最长边，防止单张图片 visual token 爆炸
            if max_edge is not None:
                _w, _h = cropped.size
                _max = max(_w, _h)
                if _max > max_edge:
                    _ratio = max_edge / _max
                    cropped = cropped.resize((int(_w * _ratio), int(_h * _ratio)), Image.Resampling.LANCZOS)

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
        import time as _time
        _t_total_start = _time.time()

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
        # STAGE 1: Planning (宏观规划 → 原子化可执行步骤)
        # 设计目标：每个步骤只定位一个视觉元素，即使是最基础的 OCR/grounding
        # 模型也能执行，从而可以分别诊断 planning vs grounding 的瓶颈。
        # ------------------------------------------
        _t1 = _time.time()
        prompt_s1 = (
            f"You are an expert visual reasoning agent. We need to solve the following question: <q>{original_question}</q>\n\n"
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

        s1_inputs = [
            {'type': 'text', 'value': prompt_s1},
            {'type': 'image', 'value': image_path}  # 原图路径交给父类读取
        ]

        ret, plan_text, resp = self.stage1_agent.generate_inner(s1_inputs, **kwargs)
        _t1_elapsed = _time.time() - _t1
        if ret != 0: return ret, plan_text, resp

        # ------------------------------------------
        # STAGE 2: Grounding (视觉锚定) by 122B
        # ------------------------------------------
        # prompt_s2 = (
        #     "Now, execute your plan. Scan the specific regions you identified and find the visual clues (text or values) required to solve the sub-problems. "
        #     "For each clue, output its bounding box coordinates normalized to the [0, 1000] range in the format [xmin, ymin, xmax, ymax]. "
        #     "Still, do NOT output the final answer yet."
        # )
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

        history = [
            {'role': 'user', 'content': s1_inputs},
            {'role': 'assistant', 'content': [{'type': 'text', 'value': plan_text}]},
            {'role': 'user', 'content': [{'type': 'text', 'value': prompt_s2}]}
        ]

        # ==============================================================
        # 【核心修改 2】: 调用独立的 grounding_agent 而不是 super()
        # ==============================================================
        _t2 = _time.time()
        # if self.verbose:
        #     self.logger.info("=== Switching to Qwen3.5-122B-A10B for Grounding ===")

        ret, grounding_text, resp = self.grounding_agent.generate_inner(history, **kwargs)
        # ret, grounding_text, resp = super().generate_inner(history, **kwargs)
        _log_diag(f'[Stage2 Grounding] {len(grounding_text)} chars')
        save_dir = "test_case/log"  # 你可以自定义保存的文件夹
        os.makedirs(save_dir, exist_ok=True)  # 确保文件夹存在
        log_file_path = os.path.join(save_dir, "grounding_results.txt")

        # 使用 "a" (append) 模式追加写入，指定 utf-8 编码防止中文乱码
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"========== Image: {image_path} ==========\n")
            f.write(grounding_text + "\n\n")

        json_log_file_path = "test_case/log/grounding_results.jsonl"
        os.makedirs(os.path.dirname(json_log_file_path), exist_ok=True)

        log_data = {
            "image_path": image_path,
            "original_question": original_question,
            "grounding_output": grounding_text
        }

        with open(json_log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

        _log_diag(f"Grounding 结果已保存至: {log_file_path}")

        if ret != 0: return ret, grounding_text, resp
        _t2_elapsed = _time.time() - _t2

        # ret, grounding_text, resp = super().generate_inner(history, **kwargs)
        # if ret != 0: return ret, grounding_text, resp

        # ------------------------------------------
        # STAGE 3: Review & Reasoning (微观推理)
        # ------------------------------------------
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
        else:
            # 回退：解析无标签的裸坐标（兼容旧版 grounding 输出）
            pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
            matches = re.findall(pattern, grounding_text)
            seen = set()
            for m in matches:
                bbox = tuple(int(val) for val in m)
                if bbox not in seen:
                    seen.add(bbox)
                    extracted_bboxes.append(list(bbox))
                    crop_labels.append(f"Region {len(crop_labels) + 1}")

        # 裁剪图数量上限。超出部分不丢弃，而是以纯文本描述形式传给 Stage 3，
        # 引导模型从缩略原图中自行定位这些区域，避免信息损失。
        MAX_CROPS = 12
        cropped_bboxes = extracted_bboxes
        cropped_labels = crop_labels
        overflow_labels = []
        if len(extracted_bboxes) > MAX_CROPS:
            _log_diag(f'[Stage3] bbox 数量 {len(extracted_bboxes)} 超过上限 {MAX_CROPS}，'
                      f'前 {MAX_CROPS} 个裁剪为子图，剩余 {len(extracted_bboxes) - MAX_CROPS} 个仅保留文字描述')
            cropped_bboxes = extracted_bboxes[:MAX_CROPS]
            cropped_labels = crop_labels[:MAX_CROPS]
            overflow_labels = crop_labels[MAX_CROPS:]

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
            f"3. Finally, step-by-step, answer the original question: <q>{original_question}</q>. "
            "Enclose the final answer within the tags <a> and </a> like this: <a>your answer</a>."
        )

        s3_user_content = [{'type': 'text', 'value': prompt_s3}]

        if cropped_bboxes:
            for i, bbox in enumerate(cropped_bboxes):
                # 在每个裁剪子图前插入语义标签，告知模型该子图代表什么
                label = cropped_labels[i] if i < len(cropped_labels) else f"Region {i+1}"
                s3_user_content.append({
                    'type': 'text',
                    'value': f'[Sub-image {i+1}: "{label}"]'
                })
                xmin, ymin, xmax, ymax = bbox
                p_bbox = [
                    int((xmin / 1000.0) * orig_w),
                    int((ymin / 1000.0) * orig_h),
                    int((xmax / 1000.0) * orig_w),
                    int((ymax / 1000.0) * orig_h)
                ]
                # 获取内存中的 Base64 编码，限制最长边 1024px 控制 visual token
                b64_string = self._crop_and_encode_base64(image_path, p_bbox, max_edge=1024)
                # 使用自定义类型 'image_base64'
                s3_user_content.append({'type': 'image_base64', 'value': b64_string})

            # 如果有溢出区域，以纯文本列表形式追加，引导模型从原图中自行定位
            if overflow_labels:
                _ov_list = "\n".join(f"  {i+1}. {lbl}" for i, lbl in enumerate(overflow_labels))
                s3_user_content.append({
                    'type': 'text',
                    'value': (
                        f"--- MISSING REGIONS (not cropped due to limit) ---\n"
                        f"The following {len(overflow_labels)} region(s) were also identified "
                        f"but could not be included as high-resolution crops. "
                        f"Please locate them in the ORIGINAL FULL IMAGE below:\n"
                        f"{_ov_list}"
                    )
                })
        else:
            s3_user_content.append({'type': 'text',
                                    'value': "Note: No specific coordinates were extracted. Rely on the original image below."})

        # 始终附带原图以提供全局上下文，客户端缩小到 1024px 以控制 token 数
        # Stage 3 已设置 extra_body=None，进一步避免服务端强制放大
        with Image.open(image_path).convert('RGB') as _orig_img:
            _ow, _oh = _orig_img.size
            _max_edge = max(_ow, _oh)
            if _max_edge > 1024:
                _ratio = 1024.0 / _max_edge
                _new_size = (int(_ow * _ratio), int(_oh * _ratio))
                _orig_img = _orig_img.resize(_new_size, Image.Resampling.LANCZOS)
            _buf = io.BytesIO()
            _orig_img.save(_buf, format='JPEG', quality=95)
            _orig_b64 = base64.b64encode(_buf.getvalue()).decode('utf-8')
        s3_user_content.append({'type': 'image_base64', 'value': _orig_b64})

        # ==============================================================
        # Stage 3 重建对话历史，仅保留文本上下文 + 裁剪小图(≤MAX_CROPS) + 溢出区域文字 + 缩略原图
        # 原图缩放到 1024px + extra_body=None，双重控制 token 数
        # 超出上限的 bbox 不丢弃，以文本描述形式保留，引导模型从原图自行定位
        # ==============================================================
        prompt_s1_text = (
            f"You are an expert visual reasoning agent. We need to solve the following question: <q>{original_question}</q>\n\n"
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
        # s3_history 中替换 S2 原始指令为中性陈述，避免 "DO NOT output the final answer"
        # 等强指令导致 Stage 3 模型角色混淆（16-25% 退化样本出现此问题）
        prompt_s2_neutral = (
            "Based on your plan, the following visual regions were identified. "
            "Each line pairs a semantic label (describing what the crop contains) with its bounding box coordinates:"
        )
        s3_history = [
            {'role': 'user', 'content': [{'type': 'text', 'value': prompt_s1_text}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'value': plan_text}]},
            {'role': 'user', 'content': [{'type': 'text', 'value': prompt_s2_neutral}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'value': grounding_text}]},
            {'role': 'user', 'content': s3_user_content},
        ]

        # Stage 3 的裁剪小图不应受 extra_body 的 min_pixels 约束
        _t3 = _time.time()
        saved_extra_body = self.extra_body
        self.extra_body = None
        try:
            ret, answer, resp = super().generate_inner(s3_history, **kwargs)
            _t3_elapsed = _time.time() - _t3
            _t_total = _time.time() - _t_total_start
            n_crops = len(cropped_bboxes) if cropped_bboxes else 0
            n_overflow = len(overflow_labels)
            _crop_info = f'crops={n_crops}'
            if n_overflow:
                _crop_info += f' overflow_text={n_overflow}'
            _log_diag(f'[3-Stage耗时] S1={_t1_elapsed:.0f}s | S2={_t2_elapsed:.0f}s | S3={_t3_elapsed:.0f}s | 总计={_t_total:.0f}s | {_crop_info}')
            return ret, answer, resp
        finally:
            self.extra_body = saved_extra_body