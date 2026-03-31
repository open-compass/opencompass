import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]

MINIMAX_API_BASE = 'https://api.minimax.io/v1/chat/completions'


class MiniMax(BaseAPIModel):
    """Model wrapper around MiniMax (legacy chatcompletion_pro API).

    .. deprecated::
        Use :class:`MiniMaxAPI` instead, which supports the latest
        OpenAI-compatible ``/v1/chat/completions`` endpoint and newer
        models (MiniMax-M2.7, MiniMax-M2.5, etc.).

    Documentation: https://platform.minimaxi.com/document/guides/chat-pro

    Args:
        path (str): The name of MiniMax model.
            e.g. ``abab5.5-chat``
        model_type (str): The type of the model
            e.g. ``chat``
        group_id (str): The id of group(like the org ID of group)
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
        group_id: str,
        model_type: str = 'chat',
        url:
        str = 'https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId=',
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
        self.type = model_type
        self.url = url + group_id
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
                The PromptDict should be organized in Test'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = [{
                'sender_type': 'USER',
                'sender_name': 'Test',
                'text': input
            }]
        else:
            messages = []
            for item in input:
                msg = {'text': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['sender_type'] = 'USER'
                    msg['sender_name'] = 'Test'
                elif item['role'] == 'BOT':
                    msg['sender_type'] = 'BOT'
                    msg['sender_name'] = 'MM智能助理'

                messages.append(msg)

        data = {
            'bot_setting': [{
                'bot_name':
                'MM智能助理',
                'content':
                'MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。' +
                'MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。'
            }],
            'reply_constraints': {
                'sender_type': 'BOT',
                'sender_name': 'MM智能助理'
            },
            'model':
            self.model,
            'messages':
            messages
        }
        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            try:
                raw_response = requests.request('POST',
                                                url=self.url,
                                                headers=self.headers,
                                                json=data)
                response = raw_response.json()
            except Exception as err:
                print('Request Error:{}'.format(err))
                time.sleep(3)
                continue
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
                msg = response['reply']
                # msg = response['choices']['messages']['text']
                return msg
            # sensitive content, prompt overlength, network error
            # or illegal prompt
            if (response.status_code == 1000 or response.status_code == 1001
                    or response.status_code == 1002
                    or response.status_code == 1004
                    or response.status_code == 1008
                    or response.status_code == 1013
                    or response.status_code == 1027
                    or response.status_code == 1039
                    or response.status_code == 2013):
                print(response.text)
                time.sleep(1)
                continue
            print(response)
            max_num_retries += 1

        raise RuntimeError(response.text)


class MiniMaxChatCompletionV2(BaseAPIModel):
    """Model wrapper around MiniMax ChatCompletionV2.

    .. deprecated::
        Use :class:`MiniMaxAPI` instead, which provides the same
        functionality with additional features (environment variable
        support, temperature clamping, thinking tag handling).

    Args:
        path (str): The name of MiniMax model.
        key (str): Authorization key.
        url (str): The API endpoint URL.
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
        url: str = MINIMAX_API_BASE,
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
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + key,
        }
        self.url = url
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
            msg_buffer, last_role = [], None
            for item in input:
                item['role'] = 'assistant' if item['role'] == 'BOT' else 'user'
                if item['role'] != last_role and last_role is not None:
                    messages.append({
                        'content': '\n'.join(msg_buffer),
                        'role': last_role
                    })
                    msg_buffer = []
                msg_buffer.append(item['prompt'])
                last_role = item['role']
            messages.append({
                'content': '\n'.join(msg_buffer),
                'role': last_role
            })

        data = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_out_len
        }

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
                try:
                    msg = response['choices'][0]['message']['content']
                    self.logger.debug(f'Generated: {msg}')
                    return msg
                except Exception:
                    code = response.get('base_resp', {}).get('status_code')
                    if code == 1002:
                        # rate limit
                        self.logger.debug('Rate limit, wait for 1s')
                        time.sleep(1)
                        continue
                    elif code == 1026:
                        return 'The request was rejected because new risk'
                    elif code == 1027:
                        return 'The request was rejected because high risk'
                    self.logger.debug(f'Resp 200, Error: {response}')
                    pass

            elif raw_response.status_code == 401:
                print('请求被拒绝 api_key错误')
                continue
            elif raw_response.status_code == 400:
                print(messages, response)
                print('请求失败，状态码:', raw_response)
                msg = 'The request was rejected because high risk'
                return msg
            elif raw_response.status_code == 429:
                print(messages, response)
                print('请求失败，状态码:', raw_response)
                time.sleep(5)
                continue
            else:
                print(messages, response)
                print('请求失败，状态码:', raw_response)
                time.sleep(1)

            max_num_retries += 1

        raise RuntimeError(raw_response)


class MiniMaxAPI(BaseAPIModel):
    """Model wrapper around MiniMax's OpenAI-compatible API.

    Supports the latest MiniMax models including MiniMax-M2.7,
    MiniMax-M2.5, and MiniMax-M2.5-highspeed via the
    ``/v1/chat/completions`` endpoint.

    Documentation: https://platform.minimaxi.com/document/ChatCompletion%20v2

    Args:
        path (str): The name of MiniMax model.
            e.g. ``MiniMax-M2.7``, ``MiniMax-M2.5``,
            ``MiniMax-M2.5-highspeed``
        key (str or List[str]): Authorization key(s). When set to
            ``'ENV'``, the key will be fetched from the environment
            variable ``$MINIMAX_API_KEY``. If it's a list, the keys
            will be used in round-robin manner. Defaults to ``'ENV'``.
        url (str): The API endpoint URL. Defaults to
            ``'https://api.minimax.io/v1/chat/completions'``.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 2.
        max_seq_len (int): The maximum sequence length. Defaults to
            204800 (MiniMax supports up to 204K context).
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retries if the API call fails.
            Defaults to 2.
        temperature (float, optional): Sampling temperature. MiniMax
            accepts values in [0, 1.0]. If not specified, the server
            default is used.
        think_tag (str, optional): The closing tag used to separate
            reasoning content from the final answer. Defaults to
            ``'</think>'``.
        system_prompt (str, optional): System prompt to prepend.
            Defaults to empty string.
    """

    def __init__(
        self,
        path: str = 'MiniMax-M2.7',
        key: Union[str, List[str]] = 'ENV',
        url: str = MINIMAX_API_BASE,
        query_per_second: int = 2,
        max_seq_len: int = 204800,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        temperature: Optional[float] = None,
        think_tag: str = '</think>',
        system_prompt: str = '',
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)

        if isinstance(key, str):
            if key == 'ENV':
                if 'MINIMAX_API_KEY' not in os.environ:
                    raise ValueError('MiniMax API key is not set. '
                                     'Please set the MINIMAX_API_KEY '
                                     'environment variable.')
                self.keys = os.getenv('MINIMAX_API_KEY').split(',')
            else:
                self.keys = [key]
        else:
            self.keys = key

        self.key_ctr = 0
        self.url = url
        self.model = path
        self.temperature = temperature
        self.think_tag = think_tag
        self.system_prompt = system_prompt

    def _get_headers(self) -> dict:
        """Get request headers with the next API key."""
        key = self.keys[self.key_ctr % len(self.keys)]
        self.key_ctr += 1
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {key}',
        }

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
            msg_buffer, last_role = [], None
            for item in input:
                item['role'] = 'assistant' if item['role'] == 'BOT' \
                    else 'user'
                if item['role'] != last_role and last_role is not None:
                    messages.append({
                        'content': '\n'.join(msg_buffer),
                        'role': last_role
                    })
                    msg_buffer = []
                msg_buffer.append(item['prompt'])
                last_role = item['role']
            messages.append({
                'content': '\n'.join(msg_buffer),
                'role': last_role
            })

        if self.system_prompt:
            messages.insert(0, {
                'role': 'system',
                'content': self.system_prompt
            })

        data = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_out_len,
        }

        if self.temperature is not None:
            # MiniMax accepts temperature in [0, 1.0]
            data['temperature'] = max(0, min(self.temperature, 1.0))

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            headers = self._get_headers()
            try:
                raw_response = requests.request('POST',
                                                url=self.url,
                                                headers=headers,
                                                json=data)
            except Exception as err:
                print('Request Error:{}'.format(err))
                time.sleep(2)
                continue

            try:
                response = raw_response.json()
            except Exception as err:
                print('Response Error:{}'.format(err))
                response = None
            self.release()

            if response is None:
                print('Connection error, reconnect.')
                self.wait()
                continue

            if raw_response.status_code == 200:
                try:
                    choice = response['choices'][0]['message']
                    msg = choice.get('content', '')

                    # Handle reasoning content (think tag)
                    reasoning = choice.get('reasoning_content', '')
                    if reasoning and self.think_tag:
                        msg = reasoning + self.think_tag + msg

                    # Strip inline <think>...</think> tags if present
                    if not reasoning and msg and self.think_tag:
                        msg = re.sub(r'<think>.*?</think>\s*', '', msg,
                                     flags=re.DOTALL)

                    self.logger.debug(f'Generated: {msg}')
                    return msg
                except Exception:
                    code = response.get('base_resp', {}).get('status_code')
                    if code == 1002:
                        self.logger.debug('Rate limit, wait for 1s')
                        time.sleep(1)
                        continue
                    elif code == 1026:
                        return 'The request was rejected because new risk'
                    elif code == 1027:
                        return 'The request was rejected because high risk'
                    self.logger.debug(f'Resp 200, Error: {response}')
                    pass

            elif raw_response.status_code == 401:
                print('Authentication failed: invalid API key')
                continue
            elif raw_response.status_code == 400:
                print(messages, response)
                print('Request failed, status:', raw_response.status_code)
                msg = 'The request was rejected because high risk'
                return msg
            elif raw_response.status_code == 429:
                print('Rate limited, waiting 5s...')
                time.sleep(5)
                continue
            else:
                print(messages, response)
                print('Request failed, status:', raw_response.status_code)
                time.sleep(1)

            max_num_retries += 1

        raise RuntimeError(raw_response)
