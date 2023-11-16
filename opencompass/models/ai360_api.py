import sys
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
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
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

    def flush(self):
        """Flush stdout and stderr when concurrent resources exists.

        When use multiproessing with standard io rediected to files, need to
        flush internal information for examination or log loss when system
        breaks.
        """
        if hasattr(self, 'tokens'):
            sys.stdout.flush()
            sys.stderr.flush()

    def acquire(self):
        """Acquire concurrent resources if exists.

        This behavior will fall back to wait with query_per_second if there are
        no concurrent resources.
        """
        if hasattr(self, 'tokens'):
            self.tokens.acquire()
        else:
            self.wait()

    def release(self):
        """Release concurrent resources if acquired.

        This behavior will fall back to do nothing if there are no concurrent
        resources.
        """
        if hasattr(self, 'tokens'):
            self.tokens.release()

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
            'temperature': 0.9,
            'max_tokens': 2048,
            'top_p': 0.5,
            'tok_k': 0,
            'repetition_penalty': 1.05,
            # "num_beams": 1,
            # "user": "OpenCompass"
        }

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
                # msg = json.load(response.text)
                # response
                # msg = response['text']
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
                # return ''
                continue
            print(raw_response)
            max_num_retries += 1

        raise RuntimeError(raw_response.text)
