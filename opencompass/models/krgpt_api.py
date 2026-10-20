import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class KrGPT(BaseAPIModel):
    is_api: bool = True

    def __init__(
            self,
            path: str = 'KrGPT',
            url: str = 'http://101.69.162.5:9300/v1/chat/completions',
            max_seq_len: int = 2048,
            meta_template: Optional[Dict] = None,
            retry: int = 2,
            generation_kwargs: Optional[Dict] = dict(),
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            retry=retry,
            generation_kwargs=generation_kwargs,
        )
        self.logger = get_logger()
        self.url = url
        self.generation_kwargs = generation_kwargs
        self.max_out_len = self.generation_kwargs.get('max_new_tokens', 1024)

    def generate(self, inputs: List[str], max_out_len: int,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [self.max_out_len] * len(inputs)))
        return results

    def _generate(self,
                  input: PromptType,
                  max_out_len: int,
                  temperature: float = 0.0) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)

        max_num_retries = 0
        while max_num_retries < self.retry:
            header = {'content-type': 'application/json'}

            try:
                data = dict(messages=messages)
                raw_response = requests.post(self.url,
                                             headers=header,
                                             data=json.dumps(data))
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
                continue
            try:
                return response['choices'][0]['message']['content'].strip()
            except KeyError:
                self.logger.error('Find error message in response: ',
                                  str(response))
                # if 'error' in response:
                #     if response['error']['code'] == 'rate_limit_exceeded':
                #         time.sleep(1)
                #         continue
                #     elif response['error']['code'] == 'insufficient_quota':
                #         self.invalid_keys.add(key)
                #         self.logger.warn(f'insufficient_quota key: {key}')
                #         continue

                #     self.logger.error('Find error message in response: ',
                #                       str(response['error']))
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')
