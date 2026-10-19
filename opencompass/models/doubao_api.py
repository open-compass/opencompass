from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    Ark = None

PromptType = Union[PromptList, str]


class Doubao(BaseAPIModel):
    """Model wrapper around Doubao.
    Documentation:
        https://www.volcengine.com/docs/82379/1263482

    Args:
        path (str): The name of Doubao model.
            e.g. `Doubao`
        accesskey (str): The access key
        secretkey (str): secretkey in order to obtain access_token
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """
    is_api: bool = True

    def __init__(self,
                 path: str,
                 accesskey: str,
                 secretkey: str,
                 query_per_second: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 retry: int = 2,
                 generation_kwargs: Dict = {
                     'temperature': 0.7,
                     'top_p': 0.9,
                 }):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        if not Ark:
            print('Please install related packages via'
                  ' `pip install \'volcengine-python-sdk[ark]\'`')

        self.accesskey = accesskey
        self.secretkey = secretkey
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
            input (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.

        messages
        [
                {
                    "role": "user",
                    "content": "天为什么这么蓝？"
                }, {
                    "role": "assistant",
                    "content": "因为有你"
                }, {
                    "role": "user",
                    "content": "花儿为什么这么香？"
                },
        ]
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

                messages.append(msg)

        client = Ark(ak=self.accesskey, sk=self.secretkey)

        req = {
            'model': self.path,
            'messages': messages,
        }
        req.update(self.generation_kwargs)

        def _chat(client, req):
            try:
                completion = client.chat.completions.create(**req)
                return completion.choices[0].message.content
            except Exception as e:
                print(e)
                return e

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            response = _chat(client, req)

            self.release()

            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                continue

            if isinstance(response, str):
                print(response)
                return response

            max_num_retries += 1

        raise RuntimeError(response)
