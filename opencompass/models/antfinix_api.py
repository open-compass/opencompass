import base64
import concurrent
import concurrent.futures
import hashlib
import hmac
import os
import random
import socket
import time
from typing import Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError
from urllib3.connection import HTTPConnection

try:
    from retrying import retry
except ImportError:
    retry = None

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class HTTPAdapterWithSocketOptions(HTTPAdapter):

    def __init__(self, *args, **kwargs):
        self._socket_options = HTTPConnection.default_socket_options + [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.SOL_TCP, socket.TCP_KEEPIDLE, 75),
            (socket.SOL_TCP, socket.TCP_KEEPINTVL, 30),
            (socket.SOL_TCP, socket.TCP_KEEPCNT, 120),
        ]
        super(HTTPAdapterWithSocketOptions, self).__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        if self._socket_options is not None:
            kwargs['socket_options'] = self._socket_options
        super(HTTPAdapterWithSocketOptions,
              self).init_poolmanager(*args, **kwargs)


class AntFinixAPI(BaseAPIModel):
    """Model wrapper around AntFinix Service.

    Args:
        ouput_key (str): key for prediction
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        generation_kwargs: other params
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
            self,
            path: str,
            key: str,
            url: str,
            meta_template: Optional[Dict] = None,
            query_per_second: int = 1,
            retry: int = 3,
            generation_kwargs: Dict = {},
            max_seq_len=32 * 1024,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
            generation_kwargs=generation_kwargs,
        )

        self.logger.info(f'AntFinix API Model Init path: {path} url={url}')

        self._key = key
        self._url = (url if url else 'https://fin-evaluator-gw.antgroup.com/api/v1/finEvaluator/evaluate')
        self._model = path
        self._sessions = []
        self._num = (int(os.environ.get('ANTFINIX_API_PARALLEL_NUM'))
                     if os.environ.get('ANTFINIX_API_PARALLEL_NUM') else 1)
        try:
            for _ in range(self._num):
                adapter = HTTPAdapterWithSocketOptions()
                sess = requests.Session()
                sess.mount('http://', adapter)
                sess.mount('https://', adapter)
                self._sessions.append(sess)
        except Exception as e:
            self.logger.error(f'Fail to setup the session. {e}')
            raise e

    def generate(
            self,
            inputs: Union[List[str], PromptList],
            max_out_len: int = 32 * 1024,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (Union[List[str], PromptList]):
                A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass' API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._num, ) as executor:
            future_to_m = {
                executor.submit(
                    self._generate,
                    self._sessions[i % self._num],
                    input,
                    max_out_len,
                ): i
                for i, input in enumerate(inputs)
            }
            results = [''] * len(inputs)
            for future in concurrent.futures.as_completed(future_to_m):
                m = future_to_m[future]
                resp = future.result()
                if resp and resp.status_code == 200:
                    try:
                        result = resp.json()
                    except Exception as e:
                        self.logger.error(f'Fail to inference; '
                                          f'model_name={self.path}; '
                                          f'error={e}, '
                                          f'request={inputs[m]}')
                    else:
                        if result.get('resultObj'):
                            results[m] = result.get('resultObj')
                        else:
                            self.logger.error(f'Receive invalid result. '
                                              f'result={result}; '
                                              f'request={inputs[m]}')
                else:
                    self.logger.error(f'Receive invalid response. '
                                      f'response={resp}; '
                                      f'request={inputs[m]}')
        self.flush()
        return results

    def _generate(
            self,
            sess,
            input: Union[str, PromptList],
            max_out_len: int,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass' API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                content = item['prompt']
                if not content:
                    continue
                message = {'content': content}
                if item['role'] == 'HUMAN':
                    message['role'] = 'user'
                elif item['role'] == 'BOT':
                    message['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    message['role'] = 'system'
                else:
                    message['role'] = item['role']
                messages.append(message)
        data = {
            "__entry_point__": "openai.chat.completion",
            'model': 'auto',
            'messages': messages,
            'max_tokens': max_out_len,
        }
        data.update(self.generation_kwargs)
        current_time = time.time()
        signature = self._sign(data, current_time)
        headers = {
            'Content-Type': 'application/json',
            'x-fin-e-gw-signature-appid': 'opencompass',
            'x-fin-e-gw-signature-timestamp': current_time,
            'x-fin-e-gw-signature': signature
        }
        request = {
            "source": "opencompass",
            "input": data,
            "modelCode": self._model
        }
        retry_num = 0
        while retry_num < self.retry:
            try:
                response = self._infer_result(request, headers, sess)
            except ConnectionError:
                time.sleep(random.randint(10, 30))
                retry_num += 1  # retry
                continue
            if response.status_code == 200:
                break  # success
            elif response.status_code == 426:
                retry_num += 1  # retry
            elif response.status_code in [302, 429, 500, 504]:
                time.sleep(random.randint(10, 30))
                retry_num += 1  # retry
            else:
                raise ValueError(f'Status code = {response.status_code}')
        else:
            # Exceed the maximal retry times.
            return ''
        return response

    def _sign(self, data, current_time):
        data_str = "postBodyForSign=" + data + "^_^" + "opencompass" + "^_^" + current_time
        data_hmac = hmac.new(self._key.encode('utf-8'), data_str.encode('utf-8'), hashlib.sha256)
        signature = base64.b64encode(data_hmac.digest())
        return signature

    def _infer_result(self, request, headers, sess):
        response = sess.request(
            'POST',
            self._url,
            json=request,
            headers=headers,
            timeout=500,
        )
        return response
