import hashlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


def get_sign(appkey, udid, timestamp, secret):
    original_str = f'{appkey}{udid}{timestamp}{secret}'
    sign = ''
    try:
        md = hashlib.sha256()
        md.update(original_str.encode('utf-8'))
        bytes_result = md.digest()
        for byte in bytes_result:
            hex_value = format(byte, '02X')
            sign += hex_value.upper()
    except Exception as e:
        print(e)
    return sign


class UniGPT(BaseAPIModel):

    def __init__(
        self,
        path: str,
        appkey: str,
        secret: str,
        url: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        temperature: float = 0.2,
    ):  # noqa E125
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
        )

        self.appkey = appkey
        self.secret = secret
        self.udid = str(uuid.uuid1())
        self.url = url
        self.model = path
        self.temperature = temperature

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

        data = {
            'model': self.path,
            'temperature': self.temperature,
            'messages': messages,
            'max_tokens': max_out_len,
        }

        timestamp = str(int(time.time()) * 1000)
        headers = {
            'appkey': self.appkey,
            'sign': get_sign(self.appkey, self.udid, timestamp, self.secret),
            'stream': 'false',
            'timestamp': timestamp,
            'udid': self.udid,
            'censor': 'none',
        }

        for _ in range(self.retry):
            try:
                response = requests.post(self.url, json=data, headers=headers)
            except Exception as e:
                print(e)
                continue
            if response is None or response.status_code != 200:
                code = response.status_code if response else -1
                print(f'request err, status_code: {code}')
                time.sleep(10)
                continue
            try:
                response = response.json()
            except Exception as e:
                print(e)
                continue
            print(response)
            if response.get('errorCode') == '8500502':
                return 'context_length_exceeded'
            return response['result']['choices'][0]['message']['content']
        raise RuntimeError(f'Failed to respond in {self.retry} retrys')
