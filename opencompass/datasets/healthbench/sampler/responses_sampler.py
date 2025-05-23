import os
import time
from typing import Any

import openai
from openai import OpenAI

from ..types import MessageList, SamplerBase, SamplerResponse


class ResponsesSampler(SamplerBase):
    """Sample from OpenAI's responses API."""

    def __init__(
        self,
        model: str = 'gpt-4.1',
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
    ):
        self.api_key_name = 'OPENAI_API_KEY'
        assert os.environ.get('OPENAI_API_KEY'), 'Please set OPENAI_API_KEY'
        self.client = OpenAI()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = 'url'
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort

    def _handle_image(
        self,
        image: str,
        encoding: str = 'base64',
        format: str = 'png',
        fovea: int = 768,
    ) -> dict[str, Any]:
        new_image = {
            'type': 'input_image',
            'image_url': f'data:image/{format};{encoding},{image}',
        }
        return new_image

    def _handle_text(self, text: str) -> dict[str, Any]:
        return {'type': 'input_text', 'text': text}

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {'role': role, 'content': content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message('developer', self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                if self.reasoning_model:
                    reasoning = ({
                        'effort': self.reasoning_effort
                    } if self.reasoning_effort else None)
                    response = self.client.responses.create(
                        model=self.model,
                        input=message_list,
                        reasoning=reasoning,
                    )
                else:
                    response = self.client.responses.create(
                        model=self.model,
                        input=message_list,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    )
                return SamplerResponse(
                    response_text=response.output_text,
                    response_metadata={'usage': response.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print('Bad Request Error', e)
                return SamplerResponse(
                    response_text='',
                    response_metadata={'usage': None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f'Rate limit exception so wait and retry {trial} after {exception_backoff} sec',
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
