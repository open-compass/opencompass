import time
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
            raw_response = requests.request('POST',
                                            url=self.url,
                                            headers=self.headers,
                                            json=data)
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
                try:
                    msg = response['choices'][0]['message']['content'].strip()
                    return msg

                except KeyError:
                    if 'error' in response:
                        # tpm(token per minitue) limit
                        if response['erro']['code'] == '1005':
                            time.sleep(1)
                            continue

                        self.logger.error('Find error message in response: ',
                                          str(response['error']))

            # sensitive content, prompt overlength, network error
            # or illegal prompt
            if (raw_response.status_code == 400
                    or raw_response.status_code == 401
                    or raw_response.status_code == 402
                    or raw_response.status_code == 429
                    or raw_response.status_code == 500):
                print(raw_response.text)
                continue
            print(raw_response)
            max_num_retries += 1

        raise RuntimeError(raw_response.text)
