# flake8: noqa

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests
from tqdm import tqdm

from opencompass.utils.prompt import PromptList

from ..base_api import BaseAPIModel
from .telechat_auth_sdk import Authorization

PromptType = Union[PromptList, str]


class TeleChatStream(BaseAPIModel):
    """
    OpenCompass adapter for TeleChat (Streaming, SSE-safe)
    """

    def __init__(
        self,
        path: str,
        url: str = '',
        key: Union[str, List[str]] = 'ENV',
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 5,
        generation_kwargs: Optional[Dict] = None,
    ):
        if generation_kwargs is None:
            generation_kwargs = {
                'temperature': 0.6,
                'top_p': 0.95,
                'repetition_penalty': 1.05,
            }

        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
            generation_kwargs=generation_kwargs,
        )

        if key == 'ENV':
            if 'TeleChat_API_KEY' not in os.environ:
                raise ValueError('TeleChat API key not set')
            key = os.getenv('TeleChat_API_KEY')

        self.app_id, self.sec_key = key.split('&&')
        self.model = path
        self.url = os.getenv('TeleChat_API_URL', url)
        self.headers = self._get_auth_headers()

    def _get_auth_headers(self) -> Dict:
        header = {
            'Content-Type': 'application/json',
            'X-APP-ID': self.app_id,
        }
        auth = Authorization()
        url_path = auth.generate_canonical_uri(self.url)
        header['Authorization'] = auth.generate_signature_all(
            self.app_id,
            self.sec_key,
            'BJ',
            str(int(time.time())),
            '259200',
            'POST',
            url_path,
            header,
        )
        return header

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
        **kwargs,
    ) -> List[str]:
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(
                        self._generate,
                        inputs,
                        [max_out_len] * len(inputs),
                    ),
                    total=len(inputs),
                    desc='Inferencing',
                ))
        self.flush()
        return results

    def _generate(self, input: PromptType, max_out_len: int) -> str:
        messages = self._build_messages(input)

        payload = {
            'model': self.model,
            'messages': messages,
            'stream': True,
            # 'max_tokens': max_out_len,
        }
        payload.update(self.generation_kwargs)

        for _ in range(self.retry):
            self.acquire()
            try:
                resp = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    timeout=60,
                )
            except Exception as e:
                self.release()
                self.logger.error(e)
                continue

            if resp.status_code != 200:
                self.release()
                self.logger.error(f'HTTP {resp.status_code}: {resp.text}')
                continue

            try:
                result = self._parse_sse_stream(resp)
                self.release()
                return result
            except Exception as e:
                self.release()
                self.logger.error(f'SSE parse failed: {e}')

        raise RuntimeError(
            f'Current issue: {payload}, has reached the maximum retry limit, but still cannot obtain the result.'
        )

    def _build_messages(self, input: PromptType) -> List[Dict]:
        if isinstance(input, str):
            return [{'role': 'user', 'content': input}]

        messages = []
        for item in input:
            role = ('user' if item['role'] == 'HUMAN' else
                    'assistant' if item['role'] == 'BOT' else 'system')
            messages.append({'role': role, 'content': item['prompt']})
        return messages

    def _parse_sse_stream(self, response: requests.Response) -> str:
        content_buf: List[str] = []
        reasoning_buf: List[str] = []

        for event in self._sse_event_iterator(response):
            for line in event.splitlines():
                line = line.strip()
                if not line.startswith('data:'):
                    continue

                data_str = line[5:].strip()
                if data_str == '[DONE]':
                    return self._merge_output(content_buf, reasoning_buf)

                try:
                    payload = json.loads(data_str)
                except Exception:
                    continue

                choices = payload.get('choices', [])
                if not choices:
                    continue

                delta = choices[0].get('delta', {})

                rc = delta.get('reasoning_content')
                if isinstance(rc, str) and rc:
                    reasoning_buf.append(rc)

                ct = delta.get('content')
                if isinstance(ct, str) and ct:
                    content_buf.append(ct)

        return self._merge_output(content_buf, reasoning_buf)

    def _sse_event_iterator(self, response: requests.Response):
        buffer = ''
        for chunk in response.iter_content(chunk_size=None,
                                           decode_unicode=True):
            if not chunk:
                continue
            buffer += chunk
            while '\n\n' in buffer:
                event, buffer = buffer.split('\n\n', 1)
                yield event
        if buffer.strip():
            yield buffer

    def _merge_output(self, content, reasoning):
        result = ''
        if reasoning:
            result += '<think>' + ''.join(reasoning) + '</think>'
        result += ''.join(content)
        return result
