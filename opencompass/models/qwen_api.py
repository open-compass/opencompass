import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class Qwen(BaseAPIModel):
    """Model wrapper around Qwen.

    Documentation:
        https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-thousand-questions/

    Args:
        path (str): The name of qwen model.
            e.g. `qwen-max`
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
                 query_per_second: int = 1,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 5,
                 generation_kwargs: Dict = {}):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        import dashscope
        dashscope.api_key = key
        self.dashscope = dashscope

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
        """
        {
          "messages": [
            {"role":"user","content":"请介绍一下你自己"},
            {"role":"assistant","content":"我是通义千问"},
            {"role":"user","content": "我在上海，周末可以去哪里玩？"},
            {"role":"assistant","content": "上海是一个充满活力和文化氛围的城市"},
            {"role":"user","content": "周末这里的天气怎么样？"}
          ]
        }

        """

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            msg_buffer, last_role = [], None
            for index, item in enumerate(input):
                if index == 0 and item['role'] == 'SYSTEM':
                    role = 'system'
                elif item['role'] == 'BOT':
                    role = 'assistant'
                else:
                    role = 'user'
                if role != last_role and last_role is not None:
                    messages.append({
                        'content': '\n'.join(msg_buffer),
                        'role': last_role
                    })
                    msg_buffer = []
                msg_buffer.append(item['prompt'])
                last_role = role
            messages.append({
                'content': '\n'.join(msg_buffer),
                'role': last_role
            })
        data = {'messages': messages}
        data.update(self.generation_kwargs)

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            try:
                response = self.dashscope.Generation.call(
                    model=self.path,
                    **data,
                )
            except Exception as err:
                print('Request Error:{}'.format(err))
                time.sleep(1)
                continue

            self.release()

            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                continue

            if response.status_code == 200:
                try:
                    msg = response.output.text
                    self.logger.debug(msg)
                    return msg
                except KeyError:
                    print(response)
                    self.logger.error(str(response.status_code))
                    time.sleep(1)
                    continue
            if response.status_code == 429:
                print(response)
                time.sleep(2)
                continue
            if response.status_code == 400:
                print('=' * 128)
                print(response)
                msg = 'Output data may contain inappropriate content.'
                return msg

            if ('Range of input length should be ' in response.message
                    or  # input too long
                    'Input data may contain inappropriate content.'
                    in response.message):  # bad input
                print(response.message)
                return ''
            print(response)
            max_num_retries += 1

        raise RuntimeError(response.message)
