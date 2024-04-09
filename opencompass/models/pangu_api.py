import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class PanGu(BaseAPIModel):
    """Model wrapper around PanGu.

    Args:
        path (str): The name of Pangu model.
            e.g. `pangu`
        access_key (str): provided access_key
        secret_key (str): secretkey in order to obtain access_token
        url (str): provide url for requests
        token_url (str): url of token server
        project_name (str): project name for generate the token
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
        access_key: str,
        secret_key: str,
        url: str,
        token_url: str,
        project_name: str,
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

        self.access_key = access_key
        self.secret_key = secret_key
        self.url = url
        self.token_url = token_url
        self.project_name = project_name
        self.model = path

        token_response = self._get_token()
        if token_response.status_code == 201:
            self.token = token_response.headers['X-Subject-Token']
            print('请求成功！')
        else:
            self.token = None
            print('token生成失败')

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

    def _get_token(self):
        url = self.token_url
        payload = {
            'auth': {
                'identity': {
                    'methods': ['hw_ak_sk'],
                    'hw_ak_sk': {
                        'access': {
                            'key': self.access_key
                        },
                        'secret': {
                            'key': self.secret_key
                        }
                    }
                },
                'scope': {
                    'project': {
                        'name': self.project_name
                    }
                }
            }
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.request('POST', url, headers=headers, json=payload)
        return response

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
                    msg['role'] = 'system'

                messages.append(msg)

        data = {'messages': messages, 'stream': False}

        # token_response = self._get_token()
        # if token_response.status_code == 201:
        #     self.token = token_response.headers['X-Subject-Token']
        #     print('请求成功！')
        # else:
        #     self.token = None
        #     print('token生成失败')

        headers = {
            'Content-Type': 'application/json',
            'X-Auth-Token': self.token
        }

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            raw_response = requests.request('POST',
                                            url=self.url,
                                            headers=headers,
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
                msg = response['choices'][0]['message']['content']
                return msg

            if (raw_response.status_code != 200):
                print(response['error_msg'])
                # return ''
                time.sleep(1)
                continue
            print(response)
            max_num_retries += 1

        raise RuntimeError(response['error_msg'])
