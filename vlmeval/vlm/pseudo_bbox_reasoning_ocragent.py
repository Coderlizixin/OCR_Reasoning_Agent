import os
import io
import re
import ast
import base64
import json
import glob as _glob
import tempfile
import pandas as pd
from PIL import Image
from vlmeval.api.gpt import OpenAIWrapper, _log_diag


class PseudoBBoxReasoningOCRAgent(OpenAIWrapper):
    """三阶段 OCR Agent（伪标签验证版）。

    与 ThreeStageOCRAgent 流程完全一致（Planning → Grounding → Review & Reasoning），
    但 Stage 2 不使用 grounding 模型动态定位，而是从 test_generate_pseudo_grounding_bbox.py
    预生成的伪标签 bbox 文件中直接读取准确的视觉锚定坐标。

    目的：验证如果给出准确的答题线索（通过答案+推理过程反向定位的 bbox），模型是否能推理正确。
    """

    def __init__(self, pseudo_bbox_file=None, trace_dir=None, padding=60, scale_factor=3.0, **kwargs):
        """
        初始化伪标签三阶段智能体。

        Args:
            pseudo_bbox_file: 伪标签 bbox 结果 TSV 文件路径（由 test_generate_pseudo_grounding_bbox.py 生成）。
                              默认: test_case/bbox_annotations/bbox_results.tsv
                              当 trace_dir 指定时，此参数被忽略。
            trace_dir: 推理轨迹 JSON 文件目录。指定后从 reasoning_traces 提取伪标签 bbox，
                       替代 TSV 文件。默认: None (使用 pseudo_bbox_file)
            padding: 裁剪时的边缘扩充像素。
            scale_factor: 局部图的放大倍数。
        """
        super().__init__(**kwargs)
        self.padding = padding
        self.scale_factor = scale_factor

        # 确保输出长度足够容纳长思维链
        if 'max_tokens' not in kwargs:
            self.max_tokens = 6144

        # 加载伪标签 bbox 数据
        if trace_dir is not None:
            self.pseudo_bbox_data = self._load_pseudo_bboxes_from_traces(trace_dir)
            _log_diag(f'[PseudoBBox] 从 reasoning traces 加载 {len(self.pseudo_bbox_data)} 条伪标签 bbox 数据')
        else:
            if pseudo_bbox_file is None:
                pseudo_bbox_file = 'test_case/bbox_annotations/bbox_results.tsv'
            self.pseudo_bbox_data = self._load_pseudo_bboxes(pseudo_bbox_file)
            _log_diag(f'[PseudoBBox] 已加载 {len(self.pseudo_bbox_data)} 条伪标签 bbox 数据')

        # Stage 1: 独立 9B 实例 (端口 22630)，避免与 Stage 3 抢占同一服务端
        self.stage1_agent = OpenAIWrapper(
            model='Qwen3.5-9B',
            api_base='http://10.3.112.34:22630/v1/chat/completions',
            key='sk-EMPTY',
            temperature=0.0,
            max_tokens=6144,
            img_size=self.img_size,
            img_detail=self.img_detail,
            extra_body=self.extra_body,
            verbose=self.verbose
        )

    def _load_pseudo_bboxes(self, file_path):
        """加载伪标签 bbox TSV 文件，构建 index → bbox 信息的查找字典。

        返回格式: {index: {'bboxes': [[xmin,ymin,xmax,ymax], ...],
                          'labels': ['desc1', 'desc2', ...]}}
        """
        if not os.path.exists(file_path):
            _log_diag(f'[PseudoBBox] 警告: 伪标签文件不存在: {file_path}')
            return {}

        try:
            df = pd.read_csv(file_path, sep='\t')
        except Exception as e:
            _log_diag(f'[PseudoBBox] 读取 TSV 失败: {e}')
            return {}

        lookup = {}
        for _, row in df.iterrows():
            if row.get('status') != 'success':
                continue

            idx = int(row['id'])
            bboxes = []
            labels = []

            # 优先从 bbox_details_stage1 解析（同时获取 desc 和 bbox_norm）
            details_str = row.get('bbox_details_stage1', '[]')
            if isinstance(details_str, str) and details_str.strip():
                try:
                    details = ast.literal_eval(details_str)
                    if isinstance(details, list):
                        for item in details:
                            if isinstance(item, dict) and 'bbox_norm' in item:
                                bbox = item['bbox_norm']
                                if len(bbox) == 4:
                                    bboxes.append([int(v) for v in bbox])
                                    labels.append(str(item.get('desc', '')))
                except (ValueError, SyntaxError):
                    pass

            # 回退：使用 extracted_bboxes（纯坐标，无描述）
            if not bboxes:
                extracted_str = row.get('extracted_bboxes', '[]')
                if isinstance(extracted_str, str) and extracted_str.strip():
                    try:
                        extracted = ast.literal_eval(extracted_str)
                        if isinstance(extracted, list):
                            for bbox in extracted:
                                if len(bbox) == 4:
                                    bboxes.append([int(v) for v in bbox])
                                    labels.append(f"Region {len(labels) + 1}")
                    except (ValueError, SyntaxError):
                        pass

            if bboxes:
                lookup[idx] = {
                    'bboxes': bboxes,
                    'labels': labels,
                }

        return lookup

    def _load_pseudo_bboxes_from_traces(self, trace_dir):
        """从 reasoning traces JSON 文件加载伪标签 bbox。

        只提取 type='read' 步骤的 bbox_norm 和 description 作为伪标签。
        返回格式与 _load_pseudo_bboxes 一致: {index: {'bboxes': [[xmin,ymin,xmax,ymax], ...],
                                                        'labels': ['desc1', 'desc2', ...]}}
        """
        lookup = {}
        if not os.path.isdir(trace_dir):
            _log_diag(f'[PseudoBBox] 警告: trace 目录不存在: {trace_dir}')
            return lookup

        json_files = _glob.glob(os.path.join(trace_dir, '*.json'))
        for fpath in json_files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    trace = json.load(f)
                idx = trace.get('index')
                if idx is None:
                    continue

                steps = trace.get('steps', [])
                # 只提取 type='read' 且有 bbox_norm 的步骤
                read_steps = [s for s in steps
                              if s.get('type') == 'read' and len(s.get('bbox_norm', [])) == 4]

                if not read_steps:
                    continue

                bboxes = []
                labels = []
                for s in read_steps:
                    bboxes.append([int(v) for v in s['bbox_norm']])
                    labels.append(str(s.get('description', f"Step {s.get('step_id', '?')}")))

                lookup[idx] = {'bboxes': bboxes, 'labels': labels}
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                _log_diag(f'[PseudoBBox] 跳过 {fpath}: {e}')

        return lookup

    @staticmethod
    def _extract_index_from_image_path(image_path):
        """从图片路径中提取数据集的 index。

        图片命名规则为 {index}.jpg（见 ImageBaseDataset.dump_image），
        如 LMUData/images/OCR_Reasoning/13.jpg → 13
        """
        basename = os.path.basename(image_path)
        # 去掉扩展名
        stem = os.path.splitext(basename)[0]
        try:
            return int(stem)
        except ValueError:
            return None

    def _crop_and_encode_base64(self, image_path, physical_bbox, max_edge=None):
        """执行物理裁剪、放大，并直接在内存中返回 Base64，绝部落盘。

        max_edge: 裁剪图最长边上限（像素），用于控制 visual token 数，None 则不限制。
        """
        with Image.open(image_path).convert('RGB') as img:
            xmin, ymin, xmax, ymax = physical_bbox
            # 防止坐标错误
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin

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
                    cropped = cropped.resize(
                        (int(_w * _ratio), int(_h * _ratio)),
                        Image.Resampling.LANCZOS
                    )

            # 纯内存转换，不产生任何临时文件
            buffer = io.BytesIO()
            cropped.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def prepare_itlist(self, inputs):
        """扩展父类的解析逻辑，支持直接传递 Base64 字符串"""
        content_list = []
        for msg in inputs:
            if msg['type'] == 'image_base64':
                # 拦截内存图，直接构造 OpenAI API 要求的格式
                img_struct = dict(
                    url=f"data:image/jpeg;base64,{msg['value']}",
                    detail=self.img_detail
                )
                content_list.append(dict(type='image_url', image_url=img_struct))
            else:
                # 普通文本或本地原图路径，原封不动地交给父类去处理
                content_list.extend(super().prepare_itlist([msg]))
        return content_list

    def generate_inner(self, inputs, **kwargs):
        """
        核心推理逻辑 — 三阶段流水线，Stage 2 替换为伪标签 bbox 查找。
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

        # 提取样本 index，用于查找伪标签 bbox
        sample_index = self._extract_index_from_image_path(image_path)
        pseudo_info = self.pseudo_bbox_data.get(sample_index, {})
        pseudo_bboxes = pseudo_info.get('bboxes', [])
        pseudo_labels = pseudo_info.get('labels', [])

        if pseudo_bboxes:
            _log_diag(f'[PseudoBBox] 样本 #{sample_index}: 命中 {len(pseudo_bboxes)} 个伪标签 bbox')
        else:
            _log_diag(f'[PseudoBBox] 样本 #{sample_index}: 无伪标签数据，将仅使用原图推理')

        # ------------------------------------------
        # STAGE 1: Planning (宏观规划 → 原子化可执行步骤)
        # 与 ThreeStageOCRAgent 完全一致
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
            {'type': 'image', 'value': image_path}
        ]

        ret, plan_text, resp = self.stage1_agent.generate_inner(s1_inputs, **kwargs)
        _t1_elapsed = _time.time() - _t1
        if ret != 0:
            return ret, plan_text, resp

        # ------------------------------------------
        # STAGE 2: Pseudo Grounding (伪标签 bbox 查找)
        # 与 ThreeStageOCRAgent 的区别：不调用 grounding 模型，
        # 而是直接从预生成的伪标签文件中读取准确的 bbox 坐标。
        # ------------------------------------------
        _t2 = _time.time()

        if pseudo_bboxes:
            # 将伪标签 bbox 格式化为与 grounding 模型输出一致的格式
            # 格式: "description" -> [xmin, ymin, xmax, ymax]
            grounding_lines = []
            for i, bbox in enumerate(pseudo_bboxes):
                label = pseudo_labels[i] if i < len(pseudo_labels) else f"Region {i + 1}"
                # 去除标签中可能存在的引号，避免破坏解析格式
                clean_label = label.replace('"', "'") if label else f"Region {i + 1}"
                grounding_lines.append(
                    f'"{clean_label}" -> [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]'
                )
            grounding_text = "\n".join(grounding_lines)
            _log_diag(f'[Stage2 PseudoGrounding] 使用 {len(pseudo_bboxes)} 个伪标签 bbox（无模型调用）')
        else:
            # 无伪标签数据时，构造一个空的 grounding 输出
            grounding_text = "No pre-computed bounding boxes available for this sample."
            _log_diag(f'[Stage2 PseudoGrounding] 无伪标签数据，跳过')

        _t2_elapsed = _time.time() - _t2

        # ------------------------------------------
        # STAGE 3: Review & Reasoning (微观推理)
        # 与 ThreeStageOCRAgent 完全一致：裁剪 bbox 区域 → 编码为子图 → 推理
        # ------------------------------------------

        # 解析 bbox（格式与 grounding 输出一致）
        labeled_pattern = r'"([^"]*)"\s*->\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        labeled_matches = re.findall(labeled_pattern, grounding_text)

        extracted_bboxes = []
        crop_labels = []
        if labeled_matches:
            for m in labeled_matches:
                desc = m[0].strip()
                bbox = [int(m[1]), int(m[2]), int(m[3]), int(m[4])]
                extracted_bboxes.append(bbox)
                crop_labels.append(desc)
        else:
            # 回退：解析无标签的裸坐标
            pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
            matches = re.findall(pattern, grounding_text)
            seen = set()
            for m in matches:
                bbox = tuple(int(val) for val in m)
                if bbox not in seen:
                    seen.add(bbox)
                    extracted_bboxes.append(list(bbox))
                    crop_labels.append(f"Region {len(crop_labels) + 1}")

        # 裁剪图数量上限
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

        # 构造 prompt_s3
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
                b64_string = self._crop_and_encode_base64(image_path, p_bbox, max_edge=1024)
                s3_user_content.append({'type': 'image_base64', 'value': b64_string})

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
            s3_user_content.append({
                'type': 'text',
                'value': "Note: No specific coordinates were extracted. Rely on the original image below."
            })

        # 始终附带原图以提供全局上下文
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

        # 重建对话历史
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

        # Stage 3: 裁剪小图不应受 extra_body 的 min_pixels 约束
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
            _src = 'pseudo' if pseudo_bboxes else 'none'
            _log_diag(f'[3-Stage耗时(伪标签)] S1={_t1_elapsed:.0f}s | S2-lookup={_t2_elapsed:.0f}s | S3={_t3_elapsed:.0f}s | 总计={_t_total:.0f}s | {_crop_info} | bbox_src={_src}')
            return ret, answer, resp
        finally:
            self.extra_body = saved_extra_body
