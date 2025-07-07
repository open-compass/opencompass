import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class ERNIEBot(BaseAPIModel):
    """Model wrapper around ERNIE-Bot.

    Documentation: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11

    Args:
        path (str): The name of ENRIE-bot model.
            e.g. `erniebot`
        model_type (str): The type of the model
            e.g. `chat`
        secretkey (str): secretkey in order to obtain access_token
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
                 secretkey: str,
                 url: str,
                 query_per_second: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 2,
                 generation_kwargs: Dict = {
                     'temperature': 0.8,
                 }):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        self.headers = {'Content_Type': 'application/json'}
        self.secretkey = secretkey
        self.key = key
        if not url.endswith('?access_token='):
            url += '?access_token='
        self.url = url
        access_token, _ = self._generate_access_token()
        self.access_token = access_token
        print(access_token)

    def _generate_access_token(self):
        try:
            BAIDU_APIKEY = self.key
            BAIDU_SECRETKEY = self.secretkey
            url = f'https://aip.baidubce.com/oauth/2.0/token?' \
                  f'client_id={BAIDU_APIKEY}&client_secret={BAIDU_SECRETKEY}' \
                  f'&grant_type=client_credentials'
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            response = requests.request('POST', url, headers=headers)
            resp_dict = response.json()
            if response.status_code == 200:
                access_token = resp_dict.get('access_token')
                refresh_token = resp_dict.get('refresh_token')
                if 'error' in resp_dict:
                    raise ValueError(f'Failed to obtain certificate.'
                                     f'{resp_dict.get("error")}')
                else:
                    return access_token, refresh_token
            else:
                error = resp_dict.get('error')
                raise ValueError(
                    f'Failed to requests obtain certificate {error}.')
        except Exception as ex:
            raise ex

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
            {"role":"assistant","content":"我是百度公司开发的人工智能语言模型"},
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
            for item in input:
                if not item['prompt']:
                    continue
                if item['role'] == 'BOT':
                    role = 'assistant'
                else:  # USER or SYSTEM
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
                raw_response = requests.request('POST',
                                                url=self.url +
                                                self.access_token,
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
                try:
                    msg = response['result']
                    self.logger.debug(msg)
                    return msg
                except KeyError:
                    print(response)
                    self.logger.error(str(response['error_code']))
                    if response['error_code'] == 336007:
                        # exceed max length
                        return ''
                    elif response['error_code'] == 336103:
                        # prompt tokens too long
                        return ''
                    else:
                        time.sleep(1)
                        continue

            if (response['error_code'] == 110 or response['error_code'] == 100
                    or response['error_code'] == 111
                    or response['error_code'] == 200
                    or response['error_code'] == 1000
                    or response['error_code'] == 1001
                    or response['error_code'] == 1002
                    or response['error_code'] == 21002
                    or response['error_code'] == 216100
                    or response['error_code'] == 336001
                    or response['error_code'] == 336003
                    or response['error_code'] == 336000
                    or response['error_code'] == 336007):
                print(response['error_msg'])
                return ''
            print(response)
            max_num_retries += 1

        raise RuntimeError(response['error_msg'])
