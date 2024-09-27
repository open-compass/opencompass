import concurrent
import concurrent.futures
import os
import socket
import traceback
from typing import Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
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


class BailingAPI(BaseAPIModel):
    """Model wrapper around Bailing Service.

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
        token: str,
        url: str,
        meta_template: Optional[Dict] = None,
        query_per_second: int = 1,
        retry: int = 3,
        generation_kwargs: Dict = {},
        max_seq_len=4096,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
            generation_kwargs=generation_kwargs,
        )

        self.logger.info(f'Bailing API Model Init path: {path} url={url}')
        if not token:
            token = os.environ.get('BAILING_API_KEY')
            if token:
                self._headers = {'Authorization': f'Bearer {token}'}
            else:
                raise RuntimeError('There is not valid token.')
        else:
            self._headers = {'Authorization': f'Bearer {token}'}

        self._headers['Content-Type'] = 'application/json'
        self._url = url if url else \
            'https://bailingchat.alipay.com/chat/completions'
        self._model = path
        self._sessions = []
        self._num = (int(os.environ.get('BAILING_API_PARALLEL_NUM'))
                     if os.environ.get('BAILING_API_PARALLEL_NUM') else 1)
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
        max_out_len: int = 4096,
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
            results = []
            for future in concurrent.futures.as_completed(future_to_m):
                m = future_to_m[future]  # noqa F841
                resp = future.result()
                if resp and resp.status_code == 200:
                    try:
                        result = resp.json()
                    except Exception as e:  # noqa F841
                        results.append('')
                    else:
                        if (result.get('choices')
                                and result['choices'][0].get('message')
                                and result['choices'][0]['message'].get(
                                    'content')):
                            results.append(
                                result['choices'][0]['message']['content'])
                else:
                    results.append('')
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
        request = {
            'model':
            self._model,
            'messages':
            messages,
            'max_seq_len':
            max(
                max_out_len if max_out_len else 4096,
                self.max_seq_len if self.max_seq_len else 4096,
            ),
        }
        request.update(self.generation_kwargs)
        try:
            retry_num = 0
            while retry_num < self.retry:
                response = self._infer_result(request, sess)
                if response.status_code == 200:
                    break  # success
                elif response.status_code == 426:
                    retry_num += 1  # retry
                else:
                    raise ValueError(f'Status code = {response.status_code}')
            else:
                raise ValueError(
                    f'Exceed the maximal retry times. Last status code '
                    f'= {response.status_code}')
        except Exception as e:
            self.logger.error(f'Fail to inference request={request}; '
                              f'model_name={self.path};  error={e}, '
                              f'stack:{traceback.format_exc()}')
            raise e
        return response

    # @retry(stop_max_attempt_number=3, wait_fixed=16000)  # ms
    def _infer_result(self, request, sess):
        response = sess.request(
            'POST',
            self._url,
            json=request,
            headers=self._headers,
            timeout=500,
        )
        return response
