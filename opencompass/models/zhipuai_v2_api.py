import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

try:
    from zhipuai.core._errors import APIStatusError, APITimeoutError
except ImportError:
    APIStatusError = None
    APITimeoutError = None

PromptType = Union[PromptList, str]


class ZhiPuV2AI(BaseAPIModel):
    """Model wrapper around ZhiPuAI.

    Args:
        path (str): The name of OpenAI's model.
        key (str): Authorization key.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(self,
                 path: str,
                 key: str,
                 query_per_second: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 2,
                 generation_kwargs: Dict = {
                     'tools': [{
                         'type': 'web_search',
                         'enable': False
                     }]
                 }):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        from zhipuai import ZhipuAI

        # self.zhipuai = zhipuai
        self.client = ZhipuAI(api_key=key)
        self.model = path

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
                if not item['prompt']:
                    continue
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)

        data = {'model': self.model, 'messages': messages}
        data.update(self.generation_kwargs)

        from pprint import pprint
        print('-' * 128)
        pprint(data)
        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()

            response = None
            from httpx import ProxyError

            try:
                response = self.client.chat.completions.create(**data)
            except APIStatusError as err:
                err_message = str(err.response.json()['error']['message'])
                status_code = str(err.status_code)
                err_code = str(err.response.json()['error']['code'])
                print('Error message:{}'.format(err_message))
                print('Statues code:{}'.format(status_code))
                print('Error code:{}'.format(err_code))

                if err_code == '1301':
                    return 'Sensitive content'
                elif err_code == '1302':
                    print('Reach rate limit')
                    time.sleep(1)
                    continue
            except ProxyError as err:
                print('Proxy Error, try again. {}'.format(err))
                time.sleep(3)
                continue
            except APITimeoutError as err:
                print('APITimeoutError {}'.format(err))
                time.sleep(3)
                continue

            self.release()

            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                max_num_retries += 1
                continue

            # if response['code'] == 200 and response['success']:
            #     msg = response['data']['choices'][0]['content']
            else:
                msg = response.choices[0].message.content
                print('=' * 128)
                print(msg)
                return msg
            # sensitive content, prompt overlength, network error
            # or illegal prompt
            if (response['code'] == 1301 or response['code'] == 1261
                    or response['code'] == 1234 or response['code'] == 1214):
                print(response['msg'])
                return ''
            print(response)
            max_num_retries += 1

        raise RuntimeError(response['msg'])
