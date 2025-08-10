from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class Mistral(BaseAPIModel):

    def __init__(
        self,
        path: str,
        api_key: str,
        url: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
    ):  # noqa E125
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
        )

        self.api_key = api_key
        self.url = url
        self.model = path

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
        self.flush()
        return results

    def _generate(self, input: PromptType, max_out_len: int = 512) -> str:
        """Generate results given an input.

        Args:
            inputs (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

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
            messages[-1]['role'] = 'user'

        data = {
            'model': self.path,
            'messages': messages,
        }

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

        from pprint import pprint
        print('-' * 128)
        pprint(data)

        for _ in range(self.retry):
            try:
                response = requests.post(self.url, json=data, headers=headers)
            except Exception as e:
                print(e)
                continue
            try:
                response = response.json()
            except Exception as e:
                print(e)
                continue
            print('=' * 128)
            pprint(response)
            try:
                msg = response['choices'][0]['message']['content']
            except Exception as e:
                print(e)
                continue
            return msg

        raise RuntimeError(f'Failed to respond in {self.retry} retrys')
