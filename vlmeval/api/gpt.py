from ..smp import *
import os
import sys
from .base import BaseAPI


def _log_diag(msg):
    """将诊断信息写入文件，避免被模型输出淹没"""
    log_dir = 'test_case/log'
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'pipeline.log'), 'a', encoding='utf-8') as f:
        import time
        f.write(f'[{time.strftime("%H:%M:%S")}] {msg}\n')

APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 900,
                 api_base: str = None,
                 max_tokens: int = 6144,
                 # img_size: int = 512,
                 # modify: 图像缩放调整
                 img_size: int = 2048,
                 img_detail: str = 'low',
                 use_azure: bool = False,
                 extra_body: dict = None,
                 **kwargs):

        self.model = model
        self.extra_body = extra_body
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure

        if 'step' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'yi-vision' in model:
            env_key = os.environ.get('YI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'internvl2-pro' in model:
            env_key = os.environ.get('InternVL2_PRO_KEY', '')
            if key is None:
                key = env_key
        elif 'abab' in model:
            env_key = os.environ.get('MiniMax_API_KEY', '')
            if key is None:
                key = env_key
        elif 'moonshot' in model:
            env_key = os.environ.get('MOONSHOT_API_KEY', '')
            if key is None:
                key = env_key
        elif 'grok' in model:
            env_key = os.environ.get('XAI_API_KEY', '')
            if key is None:
                key = env_key
        else:
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                assert env_key is not None, 'Please set the environment variable AZURE_OPENAI_API_KEY. '

                if key is None:
                    key = env_key
                assert isinstance(key, str), (
                    'Please set the environment variable AZURE_OPENAI_API_KEY to your openai key. '
                )
            else:
                env_key = os.environ.get('OPENAI_API_KEY', '')
                if key is None:
                    key = env_key
                # modify: 不检查'sk-'
                # assert isinstance(key, str) and key.startswith('sk-'), (
                #     f'Illegal openai_key {key}. '
                #     'Please set the environment variable OPENAI_API_KEY to your openai key. '
                # )
                assert isinstance(key, str), (
                    f"Illegal API key {key}. Please provide a valid string."
                )

        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout
        self.o1_model = 'o1' in model or 'o3' in model

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if use_azure:
            api_base_template = (
                '{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}'
            )
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
            assert endpoint is not None, 'Please set the environment variable AZURE_OPENAI_ENDPOINT. '
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', None)
            assert deployment_name is not None, 'Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME. '
            api_version = os.getenv('OPENAI_API_VERSION', None)
            assert api_version is not None, 'Please set the environment variable OPENAI_API_VERSION. '

            self.api_base = api_base_template.format(
                endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                api_version=os.getenv('OPENAI_API_VERSION')
            )
        else:
            if api_base is None:
                if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                    self.logger.info('Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE']
                else:
                    api_base = 'OFFICIAL'

            assert api_base is not None

            if api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif api_base.startswith('http'):
                self.api_base = api_base
            else:
                self.logger.error('Unknown API Base. ')
                raise NotImplementedError

        self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    orig_w, orig_h = img.size
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    b64_kb = len(b64) / 1024
                    # print(f'[图像诊断] 原图: {orig_w}×{orig_h}px | img_size={self.img_size} | base64: {b64_kb:.1f}KB')
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
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
        import time as _time
        _t_start = _time.time()

        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        # 保护：防止 max_tokens 变为负数（vLLM 会返回 400）
        if max_tokens is not None and max_tokens <= 0:
            _log_diag(f'[WARNING] max_tokens={max_tokens} 异常! 回退到 {self.max_tokens}. kwargs keys: {list(kwargs.keys())}')
            max_tokens = self.max_tokens

        # --- 统计请求中的图像数量 ---
        n_images = 0
        for msg in input_msgs:
            content = msg.get('content', [])
            if isinstance(content, list):
                n_images += sum(1 for item in content if item.get('type') == 'image_url')

        # Will send request if use Azure, dk how to use openai client for it
        if self.use_azure:
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
        elif 'internvl2-pro' in self.model:
            headers = {'Content-Type': 'application/json', 'Authorization': self.key}
        else:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            # max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            **kwargs)

        # 如果配置了 extra_body，合并进 payload（用于 vLLM 的 mm_processor_kwargs 等参数）
        if self.extra_body:
            payload.update(self.extra_body)

        if self.o1_model:
            payload['max_completion_tokens'] = max_tokens
            payload.pop('temperature')
        else:
            payload['max_tokens'] = max_tokens

        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        _t_elapsed = _time.time() - _t_start
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        prompt_tokens = completion_tokens = total_tokens = None
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()

            usage = resp_struct.get('usage', {})
            if usage:
                prompt_tokens = usage.get('prompt_tokens')
                completion_tokens = usage.get('completion_tokens')
                total_tokens = usage.get('total_tokens')

            print(answer)
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)

        # --- 运行统计：每次请求记录一行 JSONL ---
        import os as _os, json as _json
        _log_dir = 'test_case/log'
        _os.makedirs(_log_dir, exist_ok=True)
        _stats = {
            'timestamp': _time.strftime('%Y%m%d_%H%M%S'),
            'model': self.model,
            'api_base': self.api_base.rsplit(':', 1)[-1].split('/')[0],  # 只保留端口号
            'ret_code': ret_code,
            'elapsed_s': round(_t_elapsed, 1),
            'n_images': n_images,
            'max_tokens': max_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
        }
        if ret_code != 0:
            _stats['error'] = response.text[:500]
        with open(_os.path.join(_log_dir, 'run_stats.jsonl'), 'a', encoding='utf-8') as _f:
            _f.write(_json.dumps(_stats, ensure_ascii=False) + '\n')

        # 简短状态行 → 写入 pipeline.log
        _status = 'OK' if ret_code == 0 else f'ERR{ret_code}'
        _tok_str = f'prompt={prompt_tokens} comp={completion_tokens}' if prompt_tokens else 'tok=N/A'
        _log_diag(f'[请求] {_status} | {self.model} @ :{_stats["api_base"]} | {_t_elapsed:.0f}s | imgs={n_images} | {_tok_str}')

        # 记录失败请求的完整 payload 和响应，便于诊断 400/500 等错误
        # 每次失败都保存为独立时间戳文件，追加不覆盖
        if ret_code != 0:
            import copy, os, json as _json, time
            err_log = copy.deepcopy(payload)
            # 截断 base64 图片数据，避免日志文件过大
            for msg in err_log.get('messages', []):
                content = msg.get('content', [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            url = item.get('image_url', {}).get('url', '')
                            if len(url) > 200:
                                item['image_url']['url'] = url[:100] + f'...[截断, 总长{len(url)}字符]...' + url[-50:]
            err_log['_response_status'] = ret_code
            err_log['_response_text'] = response.text[:2000]
            err_log['_model'] = self.model
            err_log['_api_base'] = self.api_base
            err_log['_timestamp'] = time.strftime('%Y%m%d_%H%M%S')
            log_dir = 'test_case/log'
            os.makedirs(log_dir, exist_ok=True)
            # 独立文件名，按时间戳区分，不覆盖历史错误
            err_path = os.path.join(log_dir, f'failed_request_{err_log["_timestamp"]}_HTTP{ret_code}.json')
            with open(err_path, 'w', encoding='utf-8') as f:
                _json.dump(err_log, f, ensure_ascii=False, indent=2)
            _log_diag(f'[请求失败] 状态码={ret_code} | 完整请求已保存到 {err_path}')

        return ret_code, answer, response

    def get_image_token_len(self, img_path, detail='low'):
        import math
        if detail == 'low':
            return 85

        im = Image.open(img_path)
        height, width = im.size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024

        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        return total

    def get_token_len(self, inputs) -> int:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception as err:
            if 'gpt' in self.model.lower():
                if self.verbose:
                    self.logger.warning(f'{type(err)}: {err}')
                enc = tiktoken.encoding_for_model('gpt-4')
            else:
                return 0
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if 'role' in item:
                tot += self.get_token_len(item['content'])
            elif item['type'] == 'text':
                tot += len(enc.encode(item['value']))
            elif item['type'] == 'image':
                tot += self.get_image_token_len(item['value'], detail=self.img_detail)
        return tot


class GPT4V(OpenAIWrapper):

    def generate(self, message, dataset=None):
        return super(GPT4V, self).generate(message)
