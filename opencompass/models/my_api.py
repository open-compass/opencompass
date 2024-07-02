import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class MyAPIModel(BaseAPIModel):
    """Model wrapper around Baichuan.

    Documentation: https://platform.baichuan-ai.com/docs/api

    Args:
        path (str): The name of Baichuan model.
            e.g. `Baichuan2-53B`
        api_key (str): Provided api key
        url (str): Provide url
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
        api_key: str,
        url: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        generation_kwargs: Dict = {
            'temperature': 0.3,
            'top_p': 0.85,
            'top_k': 5,
            'with_search_enhance': False,
            'stream': False,
            'do_sample':False,
        }):  # noqa E125
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        
        self.api_key = api_key
        self.url = url
        self.model = path
        
    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
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
        input: str or PromptList,
        max_out_len: int = 512,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (str or PromptList): A string or PromptDict.
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

                messages.append(msg)

        data = {'model': self.model, 'messages': messages}
        data.update(self.generation_kwargs)
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': self.api_key,
        }

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            try:
                raw_response = requests.request('POST',
                                                url=self.url,
                                                headers=headers,
                                                json=data)
                response = raw_response.json()
            except Exception as err:
                print('Request Error:{}'.format(err))
                time.sleep(3)
                continue

            self.release()
            # print(response.keys())
            # print(response['choices'][0]['message']['content'])
            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                continue
            if raw_response.status_code == 200:

                # msg = response['choices'][0]['message']['content']
                msg = response
                return msg

            if raw_response.status_code != 200:
                print(raw_response.json())
                time.sleep(1)
                continue
            print(response)
            max_num_retries += 1

        raise RuntimeError(response)
