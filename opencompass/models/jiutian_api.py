import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
import requests
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class JiutianApi(BaseAPIModel):
    """Model wrapper around Jiutian API's models.

    Args:
        path (str): The name of model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        url (str): The base url
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'front','mid' and 'rear' represents the part
            of input to truncate. Defaults to 'none'.
        temperature (float, optional): What sampling temperature to use.
            If not None, will override the temperature in the `generate()`
            call. Defaults to None.
        model_id : The id of model
        appcode ： auth token
    """

    is_api: bool = True

    def __init__(self,
                 path: str = 'cmri_base',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 retry: int = 2,
                 appcode: str = '',
                 url: str = None,
                 stream: bool = True,
                 max_tokens: int = 1024,
                 model_id: str = '',
                 temperature: Optional[float] = None):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         retry=retry)
        import tiktoken
        self.tiktoken = tiktoken
        self.temperature = temperature
        self.url = url
        self.path = path
        self.stream = stream
        self.max_tokens = max_tokens
        self.model_id = model_id
        self.appcode = appcode

    def generate(
            self,
            inputs: List[str or PromptList],
            max_out_len: int = 512
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
        return results

    def _generate(self, input: str or PromptList, max_out_len: int) -> str:
        """Generate results given a list of inputs.

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
            messages = [
                {
                    "role": "user",
                    "content": input
                }
            ]
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
            messages = []

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()
            header = {
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % self.appcode
            }
            data = {
                'model': self.model_id,
                'messages': messages,
                'max_tokens': self.max_tokens,
                'stream': True
            }

            try:
                raw_response = requests.request('POST',
                                                url=self.url,
                                                headers=header,
                                                json=data,
                                                stream=True)
            except Exception as err:
                self.logger.error('Request Error:{}'.format(err))
                time.sleep(2)
                continue

            try:
                response = self.parse_event_data(raw_response)
            except Exception as err:
                self.logger.error('Response Error:{}'.format(err))
                response = None
            self.release()

            if response is None:
                self.logger.error('Connection error, reconnect.')
                self.wait()
                continue
            if isinstance(response, str):
                self.logger.error('Get stram result error, retry.')
                self.wait()
                continue
            try:
                msg = response['full_text']
                self.logger.debug(f'Generated: {msg}')
                return msg
            except:
                return ''

            max_num_retries += 1

        raise RuntimeError('max error in max_num_retries')

    def parse_event_data(self, resp) -> Dict:
        """
            解析事件数据
            :return:
            """

        def _deal_data(data: str):
            if data.startswith("data"):
                data = data.split("data:")[-1]
                try:
                    d_data = json.loads(data)
                    if "full_text" in d_data and d_data["full_text"]:
                        self.logger.debug(f"client, request response={data}")
                        return True, d_data
                except Exception as e:
                    self.logger.error(f"client, request response={data}, error={e}")

            return False, {}

        try:
            if resp.encoding is None:
                resp.encoding = 'utf-8'
            for chunk in resp.iter_lines(decode_unicode=True):
                if chunk.startswith(("event", "ping")):
                    continue
                flag, data = _deal_data(chunk)
                if flag:
                    return data
            return ''
        except Exception as e:
            self.logger.error(f"client, get stram response error={e}")
            return "get parse_event_data error"
