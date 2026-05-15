from __future__ import annotations

import os
import warnings

from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin
import base64
from openai import OpenAI
from .base import BaseAPI


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


class Qwen2VLAPI(Qwen2VLPromptMixin, BaseAPI):
    is_api: bool = True

    def __init__(
        self,
        model: str = 'qwen-vl-max-0809',
        key: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_length=1024,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        seed=3407,
        use_custom_prompt: bool = True,
        **kwargs,
    ):
        import dashscope

        self.model = model
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )

        key = os.environ.get('DASHSCOPE_API_KEY', None) if key is None else key
        assert key is not None, (
            'Please set the API Key (obtain it here: '
            'https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)'
        )
        dashscope.api_key = key
        super().__init__(use_custom_prompt=use_custom_prompt, **kwargs)

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, inputs, **kwargs) -> str:
        import dashscope

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append(
            {'role': 'user', 'content': self._prepare_content(inputs, dataset=kwargs.get('dataset', None))}
        )
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        # generate
        generation_kwargs = self.generate_kwargs.copy()
        kwargs.pop('dataset', None)
        generation_kwargs.update(kwargs)
        try:
            response = dashscope.MultiModalConversation.call(
                model=self.model,
                messages=messages,
                **generation_kwargs,
            )
            if self.verbose:
                print(response)
            answer = response.output.choices[0]['message']['content'][0]['text']
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(f'The input messages are {inputs}.')
            return -1, '', ''


class QwenVLWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'qwen-vl-plus',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 2048,
                 proxy: str = None,
                 **kwargs):

        assert model in ['qwen-vl-plus', 'qwen-vl-max']
        self.model = model
        import dashscope
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        if key is None:
            key = os.environ.get('DASHSCOPE_API_KEY', None)
        assert key is not None, (
            'Please set the API Key (obtain it here: '
            'https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)'
        )
        dashscope.api_key = key
        if proxy is not None:
            proxy_set(proxy)
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(text=msg['value']))
                elif msg['type'] == 'image':
                    content_list.append(dict(image='file://' + msg['value']))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        from dashscope import MultiModalConversation
        assert isinstance(inputs, str) or isinstance(inputs, list)

        if 'type' in inputs[0]:
            pure_text = np.all([x['type'] == 'text' for x in inputs])
        else:
            pure_text = True
            for inp in inputs:
                if not np.all([x['type'] == 'text' for x in inp['content']]):
                    pure_text = False
                    break

        assert not pure_text
        messages = self.prepare_inputs(inputs)
        gen_config = dict(max_output_tokens=self.max_tokens, temperature=self.temperature)
        gen_config.update(kwargs)
        try:
            response = MultiModalConversation.call(model=self.model, messages=messages)
            if self.verbose:
                print(response)
            answer = response.output.choices[0]['message']['content'][0]['text']
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(f'The input messages are {inputs}.')

            return -1, '', ''


class QwenVLAPI(QwenVLWrapper):

    def generate(self, message, dataset=None):
        return super(QwenVLAPI, self).generate(message)


def encode_image(image_path, max_size=2048):
    # 打开本地图片
    with Image.open(image_path) as img:
        # 如果宽或高超过了限制，则进行等比例缩放
        if img.width > max_size or img.height > max_size:
            # thumbnail 方法会自动保持长宽比例进行缩放
            img.thumbnail((max_size, max_size))
            print(f"[resize] 图片过大，已自动缩放为: {img.width}x{img.height}")

        # 将图片保存到内存中的字节缓冲区
        buffer = io.BytesIO()
        # 为了兼容性并减小传输体积，统一转换为 RGB 模式并以 JPEG 格式输出
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG")

        # 返回 Base64 编码
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


class ModelScopeQwenAPI(BaseAPI):
    """专门为 ModelScope 平台定制的 Qwen 接口调用类"""

    def __init__(self, model='Qwen/Qwen3.5-122B-A10B', key=None, api_base='https://api-inference.modelscope.cn/v1',
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        # 初始化 OpenAI 客户端 (这就是你脚本里的第一步)
        self.client = OpenAI(api_key=key, base_url=api_base)
        self.dump_image = False

    def set_dump_image(self, dump_image):
        self.dump_image = dump_image

    def generate_inner(self, inputs, **kwargs):
        # 组装 messages content 列表 (这就对应你脚本里处理图文混排的逻辑)
        content_list = []
        for item in inputs:
            if item['type'] == 'text':
                content_list.append({'type': 'text', 'text': item['value']})
            elif item['type'] == 'image':
                img_path = item['value']
                base64_image = encode_image(img_path)
                mime_type = "image/png" if img_path.lower().endswith('.png') else "image/jpeg"
                content_list.append({
                    'type': 'image_url',
                    'image_url': {'url': f"data:{mime_type};base64,{base64_image}"}
                })

        messages = [{'role': 'user', 'content': content_list}]

        try:
            # 发起请求 (注意：在自动化评测框架中，通常设置 stream=False 一次性获取结果，方便计分)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content

            # 【可选】拦截打印：让你在控制台实时看到模型的推理过程
            print(f"[ModelScope] {self.model} 推理结果:\n{answer}")
            print("\n")

            return 0, answer, "Succeed"

        except Exception as e:
            # 捕获并打印详细错误
            print(f"\n[ModelScope API Error]: {e}\n")
            return -1, "", str(e)