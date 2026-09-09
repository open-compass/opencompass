import os
import time
from typing import Any

import openai
from openai import OpenAI

from ..types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = 'You are a helpful assistant.'
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.'  # noqa: E501
    + '\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01')


class ChatCompletionSampler(SamplerBase):
    """Sample from OpenAI's chat completion API."""

    def __init__(
        self,
        model: str = 'gpt-3.5-turbo',
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        max_attempts: int = 5,
    ):
        self.api_key_name = 'OPENAI_API_KEY'
        self.client = OpenAI(
            base_url=os.getenv('OC_JUDGE_API_BASE'),
            api_key=os.getenv('OC_JUDGE_API_KEY'),
        )
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.image_format = 'url'

    def _handle_image(
        self,
        image: str,
        encoding: str = 'base64',
        format: str = 'png',
        fovea: int = 768,
    ):
        image_url = 'data:image/{};{},{}'.format(format, encoding, image)
        new_image = {
            'type': 'image_url',
            'image_url': {
                'url': image_url,
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {'type': 'text', 'text': text}

    def _pack_message(self, role: str, content: Any):
        return {'role': str(role), 'content': content}

    def _format_exception(self, e: Exception) -> str:
        parts = [f'type={type(e).__name__}', f'error={str(e)!r}']
        status_code = getattr(e, 'status_code', None)
        if status_code is not None:
            parts.append(f'status_code={status_code}')
        code = getattr(e, 'code', None)
        if code is not None:
            parts.append(f'code={code!r}')
        response = getattr(e, 'response', None)
        response_text = getattr(response, 'text', None)
        if response_text:
            parts.append(f'response_text={response_text[:1000]!r}')
        return ', '.join(parts)

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [self._pack_message('system', self.system_message)
                            ] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError(
                        'OpenAI API returned empty response; retrying')
                return SamplerResponse(
                    response_text=content,
                    response_metadata={'usage': response.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print(
                    'Judge request bad request: '
                    f'model={self.model}, {self._format_exception(e)}',
                    flush=True,
                )
                return SamplerResponse(
                    response_text='No response (bad request).',
                    response_metadata={'usage': None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                trial += 1
                error_msg = self._format_exception(e)
                if trial >= self.max_attempts:
                    print(
                        'Judge request failed permanently: '
                        f'model={self.model}, '
                        f'attempt={trial}/{self.max_attempts}, {error_msg}',
                        flush=True,
                    )
                    raise RuntimeError('OpenAI API judge request failed after '
                                       f'{self.max_attempts} attempts') from e
                exception_backoff = 2**(trial - 1)  # expontial back off
                print(
                    'Judge request failed, retrying: '
                    f'model={self.model}, '
                    f'attempt={trial}/{self.max_attempts}, '
                    f'backoff={exception_backoff}s, {error_msg}',
                    flush=True,
                )
                time.sleep(exception_backoff)
