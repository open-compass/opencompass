# flake8: noqa

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests
from tqdm import tqdm

from opencompass.utils.prompt import PromptList

from ..base_api import BaseAPIModel
from .telechat_auth_sdk import Authorization

PromptType = Union[PromptList, str]
import time


class TeleChat(BaseAPIModel):
    """
    Args:
        path (str): Model name
        key (str): Provide API Key
        url (str): Provided URL
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 2.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 5.
    """

    def __init__(self,
                 path: str,
                 url: str = '',
                 key: Union[str, List[str]] = 'ENV',
                 query_per_second: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 5,
                 generation_kwargs=None):
        if generation_kwargs is None:
            generation_kwargs = {
                'temperature': 0.6,
                'max_tokens': 16384,
                'top_p': 0.95,
                'repetition_penalty': 1.05,
            }
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        if isinstance(key, str):
            if key == 'ENV':
                if 'TeleChat_API_KEY' not in os.environ:
                    raise ValueError('TeleChat API key is not set.')
                self.keys = os.getenv('TeleChat_API_KEY')
            else:
                self.keys = key
        else:
            raise ValueError('TeleChat API key is error.')
        self.app_id, self.sec_key = self.keys.split('&&')
        self.model = path
        new_url = os.getenv('TeleChat_API_URL', '')
        if new_url:
            self.url = new_url
        else:
            self.url = url
        self.headers = self._get_auth_headers()

    def _get_auth_headers(self):
        header = {
            'Content-Type': 'application/json',
            'X-APP-ID': self.app_id,
        }

        auth = Authorization()
        url_path = auth.generate_canonical_uri(self.url)
        sign = auth.generate_signature_all(self.app_id, self.sec_key, 'BJ',
                                           str(int(time.time())), '259200',
                                           'POST', url_path, header)
        header['Authorization'] = sign
        return header

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512,
                 temperature: float = 0.7,
                 **kwargs) -> List[str]:
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
                tqdm(
                    executor.map(self._generate, inputs,
                                 [max_out_len] * len(inputs)),
                    total=len(inputs),
                    desc='Inferencing',
                ))
        self.flush()
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
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
            'model': self.model,
            'messages': messages,
            'stream': False,
            # 'max_tokens': max_out_len,
        }

        data.update(self.generation_kwargs)

        max_num_retries = 0
        while max_num_retries < self.retry:
            max_num_retries += 1
            self.acquire()
            try:
                raw_response = requests.request('POST',
                                                url=self.url,
                                                headers=self.headers,
                                                json=data)
            except Exception as e:
                self.release()
                self.logger.error(e)
                continue
            try:
                if raw_response.status_code != 200:
                    self.logger.error(f'Request failed with status code '
                                      f'{raw_response.status_code}, response: '
                                      f'{raw_response.content.decode()}')
                    continue
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error(f'JsonDecode error, got status code '
                                  f'{raw_response.status_code}, response: '
                                  f'{raw_response.content.decode()}')
                max_num_retries += 1
                continue
            if raw_response.status_code == 200:
                if 'code' not in response:
                    msg = ''
                    msg_content = response.get('choices', [{}])[0].get(
                        'message', {}).get('content', '')
                    msg_reason = response.get('choices', [{}])[0].get(
                        'message', {}).get('reasoning_content', '')
                    if msg_reason:
                        if len(msg_reason) > 0:
                            msg += ('<think>' + msg_reason + '</think>')
                    if msg_content:
                        if len(msg_content) > 0:
                            msg += msg_content
                    if msg:
                        if len(msg) > 0:
                            return msg
                    self.logger.error('Find error message in response: ',
                                      raw_response.text)
                    continue
                else:
                    self.logger.error('Find error message in response: ',
                                      raw_response.text)
                    continue
            if raw_response.status_code != 200:
                self.logger.error(raw_response.status_code)
                self.logger.error('Find error message in response: ',
                                  str(response))
                continue

        raise RuntimeError(
            f'Current issue: {data}, has reached the maximum retry limit, but still cannot obtain the result.'
        )
