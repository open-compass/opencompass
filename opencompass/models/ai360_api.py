from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class AI360GPT(BaseAPIModel):
    """Model wrapper around 360 GPT.

    Documentations: https://ai.360.com/platform/docs/overview

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
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
        self,
        path: str,  # model name, e.g.: 360GPT_S2_V9
        key: str,
        url: str = 'https://api.360.cn/v1/chat/completions',
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        generation_kwargs: Dict = {
            'temperature': 0.9,
            'max_tokens': 2048,
            'top_p': 0.5,
            'tok_k': 0,
            'repetition_penalty': 1.05,
        }):  # noqa E125
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        self.headers = {
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
        }
        self.model = path
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

        data = {
            'model': self.model,
            'messages': messages,
            'stream': False,
            # "user": "OpenCompass"
        }

        data.update(self.generation_kwargs)

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            # payload = json.dumps(data)
            try:
                raw_response = requests.request('POST',
                                                url=self.url,
                                                headers=self.headers,
                                                json=data)
            except Exception as e:
                self.release()
                print(e)
                max_num_retries += 1
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
                msg = response['choices'][0]['message']['content'].strip()
                self.logger.debug(f'Generated: {msg}')
                return msg

            # sensitive content, prompt overlength, network error
            # or illegal prompt
            if raw_response.status_code in [400, 401, 402, 429, 500]:
                if 'error' not in response:
                    print(raw_response.status_code)
                    print(raw_response.text)
                    continue
                print(response)
                # tpm(token per minitue) limit
                if response['error']['code'] == '1005':
                    self.logger.debug('tpm limit, ignoring')
                    continue
                elif response['error']['code'] == '1001':
                    msg = '参数错误:messages参数过长或max_tokens参数值过大'
                    self.logger.debug(f'Generated: {msg}')
                    return msg
                else:
                    print(response)

                self.logger.error('Find error message in response: ',
                                  str(response['error']))

            print(raw_response)
            max_num_retries += 1

        raise RuntimeError(raw_response.text)
