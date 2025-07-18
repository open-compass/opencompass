import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class BlueLMAPI(BaseAPIModel):
    """Model wrapper around BluelmV.

    Documentation:

    Args:
        path (str): The name of  model.
        api_key (str): Provided api key
        url (str): Provide url
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
        url: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 5,
        system_prompt: str = '',
        generation_kwargs: Optional[Dict] = None,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        self.headers = {
            'Content-Type': 'application/json',
        }
        self.url = url
        self.model = path
        self.system_prompt = system_prompt
        self.key = key
        self.stream = True

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

    def get_streaming_response(self, response: requests.Response):
        for chunk in response.iter_lines(chunk_size=4096,
                                         decode_unicode=False):
            if chunk:
                data = json.loads(chunk.decode('utf-8'))
                output = data.get('result')
                yield output

    def split_think(self, text: str) -> str:
        if '</think>' in text:
            answer = text.split('</think>')[1]
        else:
            if '<think>' in text:
                return 'Thinking mode too long to extract answer'
            return text
        return answer

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
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    pass
                messages.append(msg)

        data = {'text': messages, 'key': self.key, 'stream': self.stream}
        data.update(self.generation_kwargs)
        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            try:
                raw_response = requests.request('POST',
                                                url=self.url,
                                                headers=self.headers,
                                                json=data)
                final_text = None
                if self.stream:
                    for h in self.get_streaming_response(raw_response):
                        final_text = h
                else:
                    response = raw_response.json()
                    final_text = response.get('result', '')

            except Exception as err:
                self.logger.error(f'Request Error:{err}')
                time.sleep(1)
                continue

            if raw_response.status_code == 200:
                # msg = json.load(response.text)
                # response
                msg = self.split_think(final_text[0])
                self.logger.debug(f'Generated: {msg}')
                return msg

            else:
                self.logger.error(f'Request Error:{raw_response}')
                time.sleep(1)

            max_num_retries += 1

        raise RuntimeError(raw_response)
