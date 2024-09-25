import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel
import time
import hashlib
import uuid
import fake_useragent

PromptType = Union[PromptList, str]


class DianXin(BaseAPIModel):
    """Model wrapper around DianXin.

    Documentation:

    Args:
        path (str): The name of DianXin model.
            e.g. `DianXin-v2`
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
        apiKey: str, 
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        
        self.model = path
        self.apiKey = apiKey
        self.key = key
        self.url = url
    
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
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)

        self.timestamp = int(time.time())
        traceId = uuid.uuid4()
        sign_str = f"{self.apiKey}-{self.key}-{traceId}-{self.timestamp}"
        sign = hashlib.sha256(sign_str.encode('utf-8')).hexdigest()

        self.headers = {
            'User-Agent':f'{fake_useragent.UserAgent().random}',
            'Content-Type': 'application/json',
            'App-Sign': sign,
        }
        
        data = {'model': self.model, 'messages': messages, 'traceId':str(traceId), 'timestamp':self.timestamp}

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

            response = raw_response.json()
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
                print(response)
                msg = response['choices'][0]['message']['content']
                return msg

            if raw_response.status_code == 403:
                print('请求被拒绝 api_key错误')
                continue
            elif raw_response.status_code == 400:
                print(messages, response)
                print('请求失败，状态码:', raw_response)
                msg = 'The request was rejected because high risk'
                return msg
                time.sleep(1)
                continue
            elif raw_response.status_code == 429:
                print(messages, response)
                print('请求失败，状态码:', raw_response)
                time.sleep(5)
                continue

            max_num_retries += 1

        raise RuntimeError(raw_response)
