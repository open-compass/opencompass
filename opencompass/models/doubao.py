import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class Doubao(BaseAPIModel):

    def __init__(
        self,
        path: str,
        endpoint_id: str,
        access_key: str,
        secret_key: str,
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
        self.endpoint_id = endpoint_id
        self.access_key = access_key
        self.secret_key = secret_key
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError:
            self.logger.error(
                'To use the Doubao API, you need to install sdk with '
                '`pip3 install volcengine-python-sdk`')

        self.client = Ark(ak=self.access_key, sk=self.secret_key)

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
                else:
                    raise ValueError(f'Invalid role: {item["role"]}')
                messages.append(msg)

        data = dict(model=self.endpoint_id, messages=messages)

        for _ in range(self.retry):
            try:
                completion = self.client.chat.completions.create(**data)
            except Exception as e:
                print(e)
                time.sleep(1)
                continue

            generated = completion.choices[0].message.content
            self.logger.debug(f'Generated: {generated}')
            return completion.choices[0].message.content

        raise RuntimeError(f'Failed to respond in {self.retry} retrys')
