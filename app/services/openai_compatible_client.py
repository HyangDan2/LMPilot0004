from __future__ import annotations

import json
from typing import Callable
import requests


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @property
    def chat_url(self) -> str:
        if self.base_url.endswith('/chat/completions'):
            return self.base_url
        return f'{self.base_url}/chat/completions'

    @property
    def models_url(self) -> str:
        if self.base_url.endswith('/v1'):
            return f'{self.base_url}/models'
        if self.base_url.endswith('/chat/completions'):
            return self.base_url.replace('/chat/completions', '/models')
        return f'{self.base_url}/models'

    def _headers(self) -> dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def test_connection(self, model_name: str) -> tuple[bool, str]:
        try:
            response = requests.get(
                self.models_url,
                headers=self._headers(),
                timeout=min(self.timeout_seconds, 20),
            )
            response.raise_for_status()
            data = response.json()
            model_ids = [item.get('id', '') for item in data.get('data', []) if isinstance(item, dict)]
            if model_name and model_ids and model_name not in model_ids:
                return True, f'Connected. Available models: {", ".join(model_ids[:10])}'
            return True, 'Connected successfully.'
        except Exception as e:
            return False, f'Connection failed: {e}'

    def stream_chat(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        on_delta: Callable[[str], None],
    ) -> None:
        payload = {
            'model': model_name,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': True,
        }

        with requests.post(
            self.chat_url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith('data:'):
                    continue
                data_str = line[5:].strip()
                if data_str == '[DONE]':
                    break
                try:
                    item = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = item.get('choices', [])
                if not choices:
                    continue
                delta = choices[0].get('delta', {})
                content = delta.get('content')
                if content:
                    on_delta(content)

    def non_stream_chat(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            'model': model_name,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False,
        }
        response = requests.post(
            self.chat_url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
