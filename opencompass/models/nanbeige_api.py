import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class Nanbeige(BaseAPIModel):
    """Model wrapper around Nanbeige.

    Documentations:

    Args:
        path (str): Model name, e.g. `nanbeige-plus`
        key (str): Provide API Key
        url (str): Provided URL
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 2.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(self,
                 path: str,
                 key: str,
                 url: str = None,
                 query_per_second: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 3):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        self.headers = {
            'Authorization': 'Bearer ' + key,
            'Content-Type': 'application/json',
        }
        self.model = path
        self.url = url if url is not None \
            else 'http://stardustlm.zhipin.com/api/gpt/open/chat/send/sync'

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
            messages = [{'sender_type': 'USER', 'text': input}]
        else:
            messages = []
            for item in input:
                msg = {'text': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['sender_type'] = 'USER'
                elif item['role'] == 'BOT':
                    msg['sender_type'] = 'BOT'

                messages.append(msg)

        data = {
            'model': self.model,
            'messages': messages,
        }

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            raw_response = requests.request('POST',
                                            url=self.url,
                                            headers=self.headers,
                                            json=data)
            self.release()

            if raw_response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                continue

            if raw_response.status_code != 200:
                print('请求失败：', raw_response)
                print('失败信息：', raw_response.text)
                max_num_retries += 1
                continue

            response = raw_response.json()
            if response['stardustCode'] == 0:
                return response['reply']

            # exceed concurrency limit
            if response['stardustCode'] == 20035:
                print(response)
                time.sleep(2)
                continue

            print(response)
            max_num_retries += 1

        raise RuntimeError(raw_response.text)
