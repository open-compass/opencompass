import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class StepFun(BaseAPIModel):
    """Model wrapper around StepFun.

    Documentation:

    Args:
        path (str): The name of StepFun model.
            e.g. `moonshot-v1-32k`
        key (str): Authorization key.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
        self,
        path: str,
        key: str,
        url: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        system_prompt: str = '',
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + key,
        }
        self.url = url
        self.model = path
        self.system_prompt = system_prompt

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
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
            msg_buffer, last_role = [], None
            for item in input:
                item['role'] = 'assistant' if item['role'] == 'BOT' else 'user'
                if item['role'] != last_role and last_role is not None:
                    messages.append({
                        'content': '\n'.join(msg_buffer),
                        'role': last_role
                    })
                    msg_buffer = []
                msg_buffer.append(item['prompt'])
                last_role = item['role']
            messages.append({
                'content': '\n'.join(msg_buffer),
                'role': last_role
            })

        if self.system_prompt:
            system = {'role': 'system', 'content': self.system_prompt}
            messages.insert(0, system)

        data = {'model': self.model, 'messages': messages}

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            try:
                raw_response = requests.request('POST',
                                                url=self.url,
                                                headers=self.headers,
                                                json=data)
            except Exception as err:
                print('Request Error:{}'.format(err))
                time.sleep(2)
                continue

            try:
                response = raw_response.json()
            except Exception:
                response = None
            self.release()

            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                continue

            if raw_response.status_code == 200:
                # msg = json.load(response.text)
                # response
                msg = response['choices'][0]['message']['content']
                self.logger.debug(f'Generated: {msg}')
                return msg

            if raw_response.status_code == 400:
                print(messages, response)
                print('请求失败，状态码:', raw_response)
                msg = 'The context length exceeded'
                return msg
            elif raw_response.status_code == 403:
                print('请求被拒绝 api_key错误')
                continue
            elif raw_response.status_code == 429:
                print(messages, response)
                print('请求失败，状态码:', raw_response)
                time.sleep(5)
                continue
            elif raw_response.status_code == 451:
                print(messages, response)
                print('请求失败，状态码:', raw_response)
                msg = 'The request was rejected because high risk'
                return msg
            else:
                print(messages, response)
                print('请求失败，状态码:', raw_response)
                time.sleep(1)

            max_num_retries += 1

        raise RuntimeError(raw_response)
