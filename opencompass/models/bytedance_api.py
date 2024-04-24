import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

try:
    from volcengine.maas import ChatRole, MaasException, MaasService
except ImportError:
    ChatRole, MaasException, MaasService = None, None, None

PromptType = Union[PromptList, str]


class ByteDance(BaseAPIModel):
    """Model wrapper around ByteDance.

    Args:
        path (str): The name of ByteDance model.
            e.g. `skylark`
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
                 accesskey: str,
                 secretkey: str,
                 url: str,
                 query_per_second: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 2,
                 generation_kwargs: Dict = {
                     'temperature': 0.7,
                     'top_p': 0.9,
                     'top_k': 0,
                 }):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        if not ChatRole:
            print('Please install related packages via'
                  ' `pip install volcengine`')

        self.accesskey = accesskey
        self.secretkey = secretkey
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

        messages
        [
                {
                    "role": ChatRole.USER,
                    "content": "天为什么这么蓝？"
                }, {
                    "role": ChatRole.ASSISTANT,
                    "content": "因为有你"
                }, {
                    "role": ChatRole.USER,
                    "content": "花儿为什么这么香？"
                },
        ]
        """
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = [{'role': ChatRole.USER, 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = ChatRole.USER
                elif item['role'] == 'BOT':
                    msg['role'] = ChatRole.ASSISTANT

                messages.append(msg)

        maas = MaasService(self.url, 'cn-beijing')
        maas.set_ak(self.accesskey)
        maas.set_sk(self.secretkey)

        req = {
            'model': {
                'name': 'skylark-pro-public',
            },
            'messages': messages,
            'parameters': self.generation_kwargs
        }

        def _chat(maas, req):
            try:
                resp = maas.chat(req)
                return resp
            except MaasException as e:
                print(e)
                return e

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            response = _chat(maas, req)

            self.release()

            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                continue
            if not isinstance(response, MaasException):
                # response
                msg = response.choice.message.content
                return msg

            if isinstance(response, MaasException):
                print(response)
                time.sleep(1)
                continue

            max_num_retries += 1

        raise RuntimeError(response)
