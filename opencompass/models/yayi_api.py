import base64
import hashlib
import hmac
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


def generate_random_string(length=16):
    """生成随机串.

    :param length: 随机串长度，默认为 16
    :return: 随机串
    """
    letters = string.ascii_letters + string.digits
    rand_str = ''.join(random.choice(letters) for i in range(length))
    return rand_str


def get_current_time(format='%Y-%m-%d %H:%M:%S'):
    """获取当前时间.

    :param format: 时间格式，默认为 '%H:%M:%S'
    :return: 当前时间字符串
    """
    now = datetime.now()
    time_str = now.strftime(format)
    return time_str


def get_current_timestamp():
    """
    获取当前时间时间戳
    :return:
    """
    timestamp_str = int(round(time.time() * 1000))
    return str(timestamp_str)


def encode_base64_string(s):
    """对字符串进行 Base64 编码.

    :param s: 字符串
    :return: 编码后的字符串
    """
    encoded = base64.b64encode(s).decode()
    return encoded


def get_current_time_gmt_format():
    """
    获取当前时间的GMT 时间
    :return:
    """
    GMT_FORMAT = '%a, %d %b %Y %H:%M:%SGMT+00:00'
    now = datetime.now()
    time_str = now.strftime(GMT_FORMAT)
    return time_str


class Yayi(BaseAPIModel):
    """Model wrapper around SenseTime.

    Args:
        path (str): The name of SenseTime model.
            e.g. `nova-ptc-xl-v1`
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
        url: str,
        url_path: str,
        x_tilake_app_key: str,
        x_tilake_app_secret: str,
        x_tilake_ca_sginature_method: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        temperature: float = 0.4,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
        )

        self.url = url
        self.url_path = url_path
        self.X_TILAKE_APP_KEY = x_tilake_app_key
        self.X_TILAKE_APP_SECRET = x_tilake_app_secret
        self.X_TILAKE_CA_SGINATURE_METHOD = x_tilake_ca_sginature_method
        self.temperature = temperature
        self.model = path

    def generate_signature(self, method, accept, content_type, date, url_path):
        """生成签名.

        :param method:
        :param accept:
        :param content_type:
        :param date:
        :param url_path:
        :return:
        """
        string_to_sign = (method + '\n' + accept + '\n' + content_type + '\n' +
                          date + '\n' + url_path)
        string_to_sign = string_to_sign.encode('utf-8')
        secret_key = self.X_TILAKE_APP_SECRET.encode('utf-8')
        signature = hmac.new(secret_key, string_to_sign,
                             hashlib.sha256).digest()
        return encode_base64_string(signature)

    def generate_header(self, content_type, accept, date, signature):
        """生成请求头参数.

        :param content_type:
        :param accept:
        :return:
        """
        headers = {
            'x-tilake-app-key': self.X_TILAKE_APP_KEY,
            'x-tilake-ca-signature-method': self.X_TILAKE_CA_SGINATURE_METHOD,
            'x-tilake-ca-timestamp': get_current_timestamp(),
            'x-tilake-ca-nonce': generate_random_string(),
            'x-tilake-ca-signature': signature,
            'Date': date,
            'Content-Type': content_type,
            'Accept': accept,
        }
        return headers

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
                item['role'] = 'yayi' if item['role'] == 'BOT' else 'user'
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

        date = get_current_time_gmt_format()
        content_type = 'application/json'
        accept = '*/*'
        method = 'POST'
        data = {
            'id': '001',  # 请求id，无需修改。
            'model': self.model,
            'messages': messages,
            'max_new_tokens': max_out_len,  # max_new_tokens及以下参数可根据实际任务进行调整。
            'temperature': self.temperature,
            'presence_penalty': 0.85,
            'frequency_penalty': 0.16,
            'do_sample': True,
            'top_p': 1.0,
            'top_k': -1,
        }

        for _ in range(self.retry):
            signature_str = self.generate_signature(method=method,
                                                    accept=accept,
                                                    content_type=content_type,
                                                    date=date,
                                                    url_path=self.url_path)
            headers = self.generate_header(content_type=content_type,
                                           accept=accept,
                                           date=date,
                                           signature=signature_str)

            try:
                response = requests.post(self.url, json=data, headers=headers)
            except Exception as e:
                print(e)
                continue
            try:
                response = response.json()
            except Exception as e:
                print(e)
                continue
            print(response)
            try:
                return response['data']['choices'][0]['message']['content']
            except Exception as e:
                print(e)
                continue

        raise RuntimeError(f'Failed to respond in {self.retry} retrys')
