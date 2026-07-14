import os
import io
import re
import ast
import base64
import json
import glob
import pandas as pd
from PIL import Image
from vlmeval.api.gpt import OpenAIWrapper, _log_diag


class ReasoningTraceOCRAgent(OpenAIWrapper):
    """基于 GT Reasoning Trace 的结构化引导推理 Agent。

    支持四种 Oracle 模式（通过 oracle_mode 参数控制）：

    - oracle_mode=None (默认): 仅展示子图 + 描述，不透露 GT 值或推理步骤。
      模型自主完成"读取 → 推理 → 计算"。GT 仅用于定位关键区域。

    - oracle_mode='values': Oracle Values 模式。
      展示子图 + GT read 值作为显式文本。测试：知道正确数值后能否算对？

    - oracle_mode='reasoning': Oracle Reasoning 模式。
      展示子图 + GT read 值 + compute 步骤结构（仅描述和依赖关系，不含表达式和中间结果）。
      测试：给出完整推理框架后的理论上界。

    - oracle_mode='structure': Oracle Structure 模式。
      仅展示 compute/logical 步骤框架（描述+依赖），不给 GT 值、不展示子图。
      测试：仅有推理框架引导（无视觉辅助）能否提升性能。

    子图控制：
      - include_subimages=True (默认): 展示 bbox 裁剪子图
      - include_subimages=False: 不展示子图，仅依赖原图+文本提示

    架构：
      - 单轮：原图 + (可选)子图 + (可选)GT值 + (可选)推理结构 → 模型推理 → 最终答案
      - 模型输出完整推理过程（供 GPT-4 judge 评分）
      - GT 仅用于裁剪子图区域和 Oracle 模式的文本提示
    """

    def __init__(self, trace_dir=None, padding=60, scale_factor=3.0,
                 oracle_mode=None, include_subimages=True, **kwargs):
        """
        Args:
            trace_dir: 推理轨迹 JSON 文件目录。
                       默认: test_case/reasoning_traces/
            padding: 裁剪时的边缘扩充像素。
            scale_factor: 局部图的放大倍数。
            oracle_mode: None | 'values' | 'reasoning' | 'structure'
                        None: 仅子图+描述，不透露 GT 值
                        'values': 子图+描述+GT read 值作为文本
                        'reasoning': 子图+GT值+compute 步骤结构
                        'structure': 仅 compute/logical 步骤框架（无 GT 值、无子图）
            include_subimages: 是否包含 bbox 裁剪子图。默认 True。
        """
        super().__init__(**kwargs)
        self.padding = padding
        self.scale_factor = scale_factor
        self.oracle_mode = oracle_mode
        self.include_subimages = include_subimages

        if 'max_tokens' not in kwargs:
            self.max_tokens = 8192
        # Oracle 模式 + 默认模式都需要足够 tokens 输出完整推理
        # 9B thinking 模型的思考过程可能很长（15-25K chars）
        if self.max_tokens < 12288:
            self.max_tokens = 12288

        # 加载推理轨迹数据
        if trace_dir is None:
            trace_dir = 'test_case/reasoning_traces'
        self.trace_data = self._load_traces(trace_dir)
        _log_diag(f'[ReasoningTrace] oracle_mode={oracle_mode}, '
                  f'include_subimages={include_subimages}, '
                  f'已加载 {len(self.trace_data)} 条推理轨迹')

    def _load_traces(self, trace_dir):
        """从 JSON 文件目录加载推理轨迹。

        返回: {index: trace_dict}
        """
        lookup = {}
        if not os.path.isdir(trace_dir):
            _log_diag(f'[ReasoningTrace] 警告: 轨迹目录不存在: {trace_dir}')
            return lookup

        json_files = glob.glob(os.path.join(trace_dir, '*.json'))
        for fpath in json_files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    trace = json.load(f)
                idx = int(trace.get('index', 0))
                lookup[idx] = trace
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                _log_diag(f'[ReasoningTrace] 跳过 {fpath}: {e}')

        return lookup

    @staticmethod
    def _extract_index_from_image_path(image_path):
        """从图片路径中提取数据集的 index。"""
        basename = os.path.basename(image_path)
        stem = os.path.splitext(basename)[0]
        try:
            return int(stem)
        except ValueError:
            return None

    def _crop_and_encode_base64(self, image_path, physical_bbox, max_edge=None):
        """执行物理裁剪、放大，并返回 Base64 编码。"""
        with Image.open(image_path).convert('RGB') as img:
            xmin, ymin, xmax, ymax = physical_bbox
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            if xmin == xmax:
                xmax += 1
            if ymin == ymax:
                ymax += 1

            c_xmin = max(0, xmin - self.padding)
            c_ymin = max(0, ymin - self.padding)
            c_xmax = min(img.width, xmax + self.padding)
            c_ymax = min(img.height, ymax + self.padding)

            cropped = img.crop((c_xmin, c_ymin, c_xmax, c_ymax))

            if self.scale_factor != 1.0:
                new_size = (int(cropped.width * self.scale_factor),
                            int(cropped.height * self.scale_factor))
                cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)

            if max_edge is not None:
                _w, _h = cropped.size
                _max = max(_w, _h)
                if _max > max_edge:
                    _ratio = max_edge / _max
                    cropped = cropped.resize(
                        (int(_w * _ratio), int(_h * _ratio)),
                        Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            cropped.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def prepare_itlist(self, inputs):
        """扩展父类的解析逻辑，支持直接传递 Base64 字符串"""
        content_list = []
        for msg in inputs:
            if msg['type'] == 'image_base64':
                img_struct = dict(
                    url=f"data:image/jpeg;base64,{msg['value']}",
                    detail=self.img_detail)
                content_list.append(dict(type='image_url', image_url=img_struct))
            else:
                content_list.extend(super().prepare_itlist([msg]))
        return content_list

    # ------------------------------------------------------------------
    # 值比较工具
    # ------------------------------------------------------------------
    @staticmethod
    def _values_match(model_val, expected_val):
        """判断模型读取的值与期望值是否匹配。

        先字符串规范化比较，再数值容差比较（0.5%）。
        """
        mv = str(model_val).strip().lower().rstrip('%')
        ev = str(expected_val).strip().lower().rstrip('%')

        if mv == ev:
            return True

        # 尝试数值比较
        try:
            mn = float(mv.replace(',', '').replace(' ', ''))
            en = float(ev.replace(',', '').replace(' ', ''))
            if en == 0:
                return abs(mn) < 0.001
            return abs(mn - en) / abs(en) < 0.005  # 0.5% 容差
        except (ValueError, ZeroDivisionError):
            pass

        return False

    # ------------------------------------------------------------------
    # 核心推理逻辑
    # ------------------------------------------------------------------
    def generate_inner(self, inputs, **kwargs):
        """单轮推理流水线。根据 oracle_mode 构建不同的 prompt。

        oracle_mode=None: 仅子图+描述，模型自主读值+推理
        oracle_mode='values': 子图+描述+GT读值文本，模型只需推理计算
        oracle_mode='reasoning': 子图+GT值+compute步骤结构，模型按框架计算
        """
        import time as _time
        _t_start = _time.time()

        # 1. 提取图片路径和问题
        image_path = None
        original_question = ""
        for item in inputs:
            if item['type'] == 'image':
                image_path = item['value']
            elif item['type'] == 'text':
                original_question += item['value']

        if not image_path:
            return super().generate_inner(inputs, **kwargs)

        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        # 2. 查找推理轨迹
        sample_index = self._extract_index_from_image_path(image_path)
        trace = self.trace_data.get(sample_index, {})
        steps = trace.get('steps', [])
        read_steps = [s for s in steps if s.get('type') == 'read']
        compute_steps = [s for s in steps if s.get('type') == 'compute']
        logical_steps = [s for s in steps if s.get('type') == 'logical']

        if not read_steps:
            _log_diag(f'[ReasoningTrace] 样本 #{sample_index}: 无推理轨迹，退化为普通调用')
            return super().generate_inner(inputs, **kwargs)

        _log_diag(f'[ReasoningTrace] 样本 #{sample_index}: '
                  f'{len(read_steps)}r + {len(compute_steps)}c + {len(logical_steps)}l, '
                  f'mode={self.oracle_mode or "default"}, sub_imgs={self.include_subimages}')

        # ==================================================================
        # 构建 valid_read_steps（始终需要，用于 prompt 中的值/描述）
        # ==================================================================
        valid_read_steps = []
        for s in read_steps:
            bbox_norm = s.get('bbox_norm', [])
            if len(bbox_norm) == 4 or self.oracle_mode in ('values', 'reasoning', 'structure'):
                # Oracle 模式下即使没有 bbox 也保留（值来自 GT，不依赖子图）
                valid_read_steps.append(s)

        if not valid_read_steps:
            _log_diag(f'[ReasoningTrace] 样本 #{sample_index}: 无有效推理步骤，退化为普通调用')
            return super().generate_inner(inputs, **kwargs)

        # ==================================================================
        # 构建子图列表（仅在 include_subimages=True 时）
        # ==================================================================
        turn_content = []

        # 原图（全局上下文）
        full_b64 = self._crop_and_encode_base64(image_path, [0, 0, orig_w, orig_h],
                                                 max_edge=2048)

        sub_image_blocks = []

        if self.include_subimages:
            for s in valid_read_steps:
                sid = s['step_id']
                desc = s.get('description', f'Step {sid}')
                bbox_norm = s.get('bbox_norm', [])
                if len(bbox_norm) != 4:
                    continue

                # 物理坐标转换
                p_bbox = [
                    int((bbox_norm[0] / 1000.0) * orig_w),
                    int((bbox_norm[1] / 1000.0) * orig_h),
                    int((bbox_norm[2] / 1000.0) * orig_w),
                    int((bbox_norm[3] / 1000.0) * orig_h)
                ]
                b64 = self._crop_and_encode_base64(image_path, p_bbox, max_edge=1024)

                sub_image_blocks.append({
                    'type': 'text',
                    'value': f'--- [Region {sid}: "{desc}"] ---'
                })
                sub_image_blocks.append({'type': 'image_base64', 'value': b64})

        # ==================================================================
        # 根据 oracle_mode 构建不同的 prompt
        # ==================================================================
        if self.oracle_mode == 'values':
            prompt_text = self._build_oracle_values_prompt(
                original_question, valid_read_steps, include_subimages=self.include_subimages)
        elif self.oracle_mode == 'reasoning':
            # 合并所有步骤（read + compute + logical）为统一顺序列表
            all_steps = valid_read_steps + compute_steps + logical_steps
            prompt_text = self._build_oracle_reasoning_prompt(
                original_question, all_steps,
                include_subimages=self.include_subimages)
        elif self.oracle_mode == 'structure':
            # 合并所有步骤（read + compute + logical）为统一顺序列表
            all_steps = valid_read_steps + compute_steps + logical_steps
            prompt_text = self._build_structure_prompt(
                original_question, all_steps)
        else:
            prompt_text = self._build_default_prompt(
                original_question, valid_read_steps, include_subimages=self.include_subimages)

        turn_content.append({'type': 'image_base64', 'value': full_b64})
        turn_content.append({'type': 'text', 'value': prompt_text})
        for block in sub_image_blocks:
            turn_content.append(block)

        # ==================================================================
        # 执行单轮推理
        # ==================================================================
        ret, raw_answer, resp = super().generate_inner(turn_content, **kwargs)
        _t_elapsed = _time.time() - _t_start

        if ret != 0:
            return ret, raw_answer, resp

        # ==================================================================
        # 诊断：值覆盖率 + 答案提取日志
        # ==================================================================
        clean_answer = self._extract_final_answer(raw_answer)

        gt_read_total = 0
        gt_read_correct = 0
        for s in valid_read_steps:
            expected = str(s.get('expected_value', ''))
            if not expected:
                continue
            gt_read_total += 1
            if expected.lower() in raw_answer.lower():
                gt_read_correct += 1

        read_acc = gt_read_correct / gt_read_total if gt_read_total > 0 else 0
        _log_diag(f'[ReasoningTrace] 样本 #{sample_index}: '
                  f'耗时={_t_elapsed:.0f}s | mode={self.oracle_mode or "default"} | '
                  f'输出长度={len(str(raw_answer))} | '
                  f'答案提取={(clean_answer or "")[:60]} | '
                  f'值覆盖={gt_read_correct}/{gt_read_total}={read_acc:.1%}')

        # CRITICAL: 返回 raw_answer（完整输出），让框架的 GPT-4 judge 能
        # 看到完整推理过程并给出有意义的 reason_score。
        # clean_answer 仅用于上述诊断日志。
        return ret, raw_answer, resp

    # ------------------------------------------------------------------
    # Prompt 构建器
    # ------------------------------------------------------------------
    @staticmethod
    def _build_default_prompt(question, valid_read_steps, include_subimages=True):
        """默认模式：原图 + (可选)子图 + 描述，不透露 GT 值。"""
        if include_subimages:
            read_instruction_lines = [
                f'  Region {s["step_id"]}: "{s.get("description", "")}"'
                for s in valid_read_steps
            ]
            region_block = (
                f"=== KEY REGIONS TO READ ===\n"
                + "\n".join(read_instruction_lines) + "\n\n"
            )
            region_hint = (
                f"and {len(valid_read_steps)} zoomed-in regions that contain the values "
                f"you need.\n\n"
            )
        else:
            region_block = ""
            hint_lines = []
            for s in valid_read_steps:
                sid = s['step_id']
                desc = s.get('description', f'Step {sid}')
                hint_lines.append(f'  - {desc}')
            region_hint = (
                "Here are the key values you need to locate in the image:\n"
                + "\n".join(hint_lines)
                + "\n\n"
            )

        anti_loop_default = (
            f"CRITICAL: Read each value exactly ONCE. Do NOT re-read, re-verify, "
            f"or loop over the same data. Do NOT discuss chart colors, legends, or "
            f"visual details. Once you have the values, compute immediately.\n"
        ) if not include_subimages else ""

        return (
            f"You are solving a visual reasoning problem. Below is the full image "
            f"{region_hint}"
            f"=== QUESTION ===\n"
            f"{question}\n\n"
            f"{region_block}"
            f"=== INSTRUCTIONS ===\n"
            f"{anti_loop_default}"
            f"1. Locate and read each required value from the image.\n"
            f"2. After reading all values, plan your step-by-step calculation.\n"
            f"3. Show your work clearly — state each value, then show each "
            f"calculation exactly once.\n"
            f"4. Do NOT get stuck re-analyzing visual details. Read, compute, answer.\n"
            f"5. At the VERY END of your response, output:\n"
            f"   FINAL_ANSWER: <your answer>\n\n"
            f"Output your full reasoning, then the final answer in the format "
            f"requested by the original question."
        )

    @staticmethod
    def _build_oracle_values_prompt(question, valid_read_steps, include_subimages=True):
        """Oracle Values 模式：原图 + (可选)子图 + GT read 值作为显式文本。

        目的：消除视觉读取瓶颈，测试模型的计算/推理能力。
        """
        value_lines = []
        for s in valid_read_steps:
            val = s.get('expected_value', '?')
            unit = s.get('expected_unit', '')
            unit_str = f' ({unit})' if unit else ''
            value_lines.append(
                f'  Value {s["step_id"]}: **{val}**{unit_str}'
            )

        if include_subimages:
            read_instruction_lines = [
                f'  Region {s["step_id"]}: "{s.get("description", "")}"'
                for s in valid_read_steps
            ]
            region_block = (
                f"=== KEY REGIONS (visual reference only) ===\n"
                + "\n".join(read_instruction_lines) + "\n\n"
            )
        else:
            region_block = ""

        anti_loop = (
            f"CRITICAL: Do NOT examine or describe the image. Do NOT discuss "
            f"chart colors, legends, visual layout, or try to verify values. "
            f"This is a pure computation task — use ONLY the given values.\n"
        ) if not include_subimages else ""

        return (
            f"You are solving a visual reasoning problem. The correct values "
            f"have been pre-extracted from the image for you.\n\n"
            f"=== QUESTION ===\n"
            f"{question}\n\n"
            f"=== VERIFIED VALUES (ground truth — use directly, do NOT verify) ===\n"
            + "\n".join(value_lines) + "\n\n"
            f"{region_block}"
            f"=== INSTRUCTIONS ===\n"
            f"{anti_loop}"
            f"1. The values above are CORRECT. Do NOT check them against the image.\n"
            f"   Do NOT discuss chart colors, legends, or visual details.\n"
            f"2. Use these values directly to answer the question.\n"
            f"3. Show your calculation steps clearly.\n"
            f"4. Be concise — focus on the math, not on visual description.\n"
            f"5. At the VERY END of your response, output:\n"
            f"   FINAL_ANSWER: <your answer>\n\n"
            f"IMPORTANT: This is a computation test. The values are given.\n"
            f"Just calculate and output the answer.\n"
            f"Output the final answer in the format requested by the original question."
        )

    @staticmethod
    def _build_oracle_reasoning_prompt(question, all_steps, include_subimages=True):
        """Oracle Reasoning 模式：原图 + (可选)子图 + GT 值 + 统一顺序的 REASONING PLAN。

        展示 GT 读值和步骤框架（read/compute/logical 统一排序），
        使用 plan 文本（任务导向，不含具体数值/结论）。
        目的：测试推理框架引导的理论上界。

        all_steps: 合并后的所有步骤，按 step_id 排序。
        """
        # GT 值列表（仅 read 步骤）
        value_lines = []
        for s in all_steps:
            if s.get('type') != 'read':
                continue
            val = s.get('expected_value', '?')
            unit = s.get('expected_unit', '')
            unit_str = f' ({unit})' if unit else ''
            value_lines.append(
                f'  Step {s["step_id"]}: **{val}**{unit_str}'
            )

        # 统一顺序的 REASONING PLAN
        sequential_plan = ReasoningTraceOCRAgent._build_sequential_plan(all_steps)

        anti_loop_or = (
            f"CRITICAL: Do NOT examine or describe the image. Do NOT discuss "
            f"chart colors, legends, or visual layout. "
            f"This is a pure computation task — use ONLY the given values.\n"
        ) if not include_subimages else ""

        return (
            f"You are solving a visual reasoning problem. The correct values "
            f"have been pre-extracted and a reasoning plan is provided.\n\n"
            f"=== QUESTION ===\n{question}\n\n"
            f"=== VERIFIED VALUES (ground truth — use directly, do NOT verify) ===\n"
            + "\n".join(value_lines) + "\n\n"
            f"{sequential_plan}\n\n"
            f"=== CRITICAL RULES ===\n"
            f"{anti_loop_or}"
            f"- The verified values above are CORRECT. Do NOT check them against "
            f"the chart. Do NOT discuss chart colors, legends, or visual details.\n"
            f"- Follow the reasoning plan above in order. Execute each step.\n"
            f"- Show intermediate results for each step.\n"
            f"- Be concise — focus on the math, not on visual description.\n"
            f"- At the VERY END of your response, output:\n"
            f"  FINAL_ANSWER: <your answer>\n\n"
            f"IMPORTANT: This is a computation test. The values and plan are given.\n"
            f"Just follow the plan, compute, and output the answer.\n"
            f"Output the final answer in the format requested by the original question."
        )

    @staticmethod
    def _build_structure_prompt(question, all_steps):
        """Oracle Structure 模式：仅原图 + 统一顺序的 REASONING PLAN。

        不给 GT 值、不展示子图。模型必须自行从原图读取数值，
        但按照给定的步骤框架执行推理。目的：测试仅有推理框架
        引导（无视觉辅助）能否提升性能。

        all_steps: 合并后的所有步骤（read + compute + logical），按 step_id 排序。
        """
        if not all_steps:
            return (
                f"You are solving a visual reasoning problem.\n\n"
                f"=== QUESTION ===\n{question}\n\n"
                f"=== INSTRUCTIONS ===\n"
                f"1. Carefully examine the image to find all relevant values.\n"
                f"2. Plan and execute your calculations step by step.\n"
                f"3. Show your work clearly.\n"
                f"4. At the VERY END of your response, output:\n"
                f"   FINAL_ANSWER: <your answer>\n\n"
                f"Output the final answer in the format requested by the original question."
            )

        sequential_plan = ReasoningTraceOCRAgent._build_sequential_plan(all_steps)

        return (
            f"You are solving a visual reasoning problem. A reasoning plan is "
            f"provided to guide your analysis.\n\n"
            f"=== QUESTION ===\n{question}\n\n"
            f"{sequential_plan}\n\n"
            f"=== CRITICAL RULES ===\n"
            f"- You must READ all required values from the image yourself.\n"
            f"  No values are provided — use your vision to locate and extract them.\n"
            f"- Follow the reasoning plan above in order. Execute each step.\n"
            f"- Show your work clearly: state each value you read, then show each "
            f"calculation.\n"
            f"- Read each value ONCE. Do NOT get stuck re-reading or re-verifying "
            f"the same value. Do NOT discuss chart colors, legends, or visual "
            f"details beyond reading the needed numbers.\n"
            f"- At the VERY END of your response, output:\n"
            f"  FINAL_ANSWER: <your answer>\n\n"
            f"IMPORTANT: The plan above is a guide, not a script. Read the values "
            f"yourself, then follow the reasoning structure.\n"
            f"Output the final answer in the format requested by the original question."
        )

    @staticmethod
    def _extract_final_answer(raw_response):
        """从模型输出中提取最终答案。

        支持 FINAL_ANSWER: 标记、<a> 标签、以及末尾短文本三种策略。
        """
        text = str(raw_response)

        # Strategy 1: FINAL_ANSWER: 标记（取最后一次有意义的出现）
        matches = re.findall(r'FINAL_ANSWER\s*:\s*(.+?)(?:\n|$)', text)
        for ans in reversed(matches):
            ans = ans.strip()
            if ans and len(ans) > 1 and ans.lower() not in (
                    'your answer', 'your final answer', '...', 'and', 'n/a'):
                return ans

        # Strategy 2: 最后一个有意义的 <a> 标签
        all_a_tags = re.findall(r'<a>(.*?)</a>', text)
        meaningful = [
            a.strip() for a in all_a_tags
            if a.strip() and len(a.strip()) > 1
            and a.strip().lower() not in ('your answer', '...', 'and', '` and `')
        ]
        if meaningful:
            return meaningful[-1]

        # Strategy 3: 兜底 — 取最后 300 字符中的最后一行非空文本
        tail = text[-500:]
        lines = [l.strip() for l in tail.split('\n') if l.strip()]
        if lines:
            return lines[-1]

        return text[-500:]

    # ------------------------------------------------------------------
    # Plan 构建工具
    # ------------------------------------------------------------------
    @staticmethod
    def _get_step_plan(step):
        """获取步骤的任务导向 plan 文本。优先使用 'plan' 字段，fallback 到 'description'。"""
        return step.get('plan') or step.get('description', '')

    @staticmethod
    def _build_sequential_plan(all_steps):
        """构建统一的 REASONING PLAN section，按 step_id 顺序排列所有步骤。

        不按类型分组——read/compute/logical 步骤在同一个有序列表中，
        保留依赖关系，使用 plan 文本（不含 GT 值/结论）。
        """
        lines = ["=== REASONING PLAN (follow this sequence) ==="]
        for s in sorted(all_steps, key=lambda x: x['step_id']):
            sid = s['step_id']
            plan_text = ReasoningTraceOCRAgent._get_step_plan(s)
            deps = s.get('depends_on', [])
            if deps:
                lines.append(f'  Step {sid}: {plan_text} (using results from steps {deps})')
            else:
                lines.append(f'  Step {sid}: {plan_text}')
        return "\n".join(lines)
