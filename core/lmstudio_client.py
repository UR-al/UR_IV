"""LM Studio's OpenAI-compatible local chat API (no SDK dependency)."""
from __future__ import annotations

import base64
import json
from urllib.parse import urlsplit

import requests


class LMStudioClient:
    def __init__(self, base_url="http://localhost:1234", model=""):
        root = base_url.strip().rstrip('/')
        parsed = urlsplit(root)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('LM Studio 서버 주소는 http(s) URL이어야 합니다')
        self.base_url = root if root.endswith('/v1') else root + '/v1'
        self.model = model

    def list_models(self):
        with requests.get(self.base_url + '/models', timeout=(3, 8)) as response:
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get('data'), list):
            raise ValueError('LM Studio 모델 목록 응답 형식이 올바르지 않습니다')
        return list(dict.fromkeys(item['id'] for item in data['data']
                                 if isinstance(item, dict) and isinstance(item.get('id'), str) and item['id']))

    def get_model_info(self):
        if self.model not in self.list_models():
            raise ValueError('LM Studio 서버에서 선택한 모델을 찾지 못했습니다')
        # /v1/models does not establish architecture, vision or thinking support.
        return {'architecture': '', 'parameterSize': '', 'quantization': '',
                'moe': None, 'experts': None, 'activeExperts': None,
                'vision': None, 'thinkingMode': 'unknown', 'capabilities': []}

    @staticmethod
    def _messages(messages):
        result = []
        for message in messages:
            content = message.get('content', '')
            if message.get('images'):
                parts = [{'type': 'text', 'text': content}]
                for encoded in message['images']:
                    header = base64.b64decode(encoded[:32])
                    mime = ('image/png' if header.startswith(b'\x89PNG') else
                            'image/webp' if header.startswith(b'RIFF') else
                            'image/gif' if header.startswith(b'GIF8') else 'image/jpeg')
                    parts.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}})
                content = parts
            result.append({'role': message['role'], 'content': content})
        return result

    def chat_stream(self, messages, *, model=None, options=None, think=None, schema=None,
                    on_token=None, on_thinking=None, should_stop=None, timeout=600):
        options = options or {}
        payload = {'model': model or self.model, 'messages': self._messages(messages), 'stream': True}
        if 'temperature' in options:
            payload['temperature'] = options['temperature']
        if options.get('num_predict', -1) > 0:
            payload['max_tokens'] = options['num_predict']
        if schema is not None:
            from core.structured_output import parse_schema
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'chat_output', 'strict': True, 'schema': parse_schema(schema)}}
            payload.setdefault('max_tokens', 4096)
        # num_ctx and Ollama's think are not portable OpenAI chat options.
        pieces, thoughts = [], []
        reason, count = None, None
        stopped = lambda: should_stop is not None and should_stop()

        def result():
            return {'content': ''.join(pieces), 'thinking': ''.join(thoughts),
                    'stopped': bool(stopped()), 'done_reason': reason, 'eval_count': count}

        if stopped():
            return result()
        with requests.post(self.base_url + '/chat/completions', json=payload,
                           stream=True, timeout=(10, timeout)) as response:
            if response.status_code != 200:
                raise RuntimeError(f'LM Studio {response.status_code}: {response.text[:1000]}')
            response.encoding = 'utf-8'
            for line in response.iter_lines(decode_unicode=True):
                if stopped():
                    return result()
                if not line or not line.startswith('data:'):
                    continue
                raw = line[5:].strip()
                if raw == '[DONE]':
                    if reason is None:
                        raise RuntimeError('LM Studio가 완료 사유 없이 응답을 끝냈습니다. 받은 내용은 유지됩니다.')
                    return result()
                data = json.loads(raw)
                if data.get('error'):
                    raise RuntimeError(str(data['error'])[:1000])
                count = (data.get('usage') or {}).get('completion_tokens', count)
                for choice in data.get('choices') or []:
                    if choice.get('index', 0) != 0:
                        continue
                    delta = choice.get('delta') or {}
                    for key, target, callback in [('content', pieces, on_token), ('reasoning_content', thoughts, on_thinking)]:
                        piece = delta.get(key)
                        if isinstance(piece, str) and piece:
                            target.append(piece)
                            if callback:
                                callback(piece)
                    if choice.get('finish_reason'):
                        reason = choice['finish_reason']
        if stopped():
            return result()
        raise RuntimeError('LM Studio 응답 완료 신호 전에 연결이 끊어졌습니다. 받은 내용은 유지됩니다.')
