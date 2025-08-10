import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class SenseTime(BaseAPIModel):
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
        key: str = 'ENV',
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        parameters: Optional[Dict] = None,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)

        if isinstance(key, str):
            self.keys = os.getenv('SENSENOVA_API_KEY') if key == 'ENV' else key
        else:
            self.keys = key

        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.keys}'
        }
        self.url = url
        self.model = path
        self.params = parameters

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
                if not item['prompt']:
                    continue
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

        data = {'messages': messages, 'model': self.model}
        if self.params is not None:
            data.update(self.params)

        stream = data['stream']

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()

            max_num_retries += 1
            try:
                raw_response = requests.request('POST',
                                                url=self.url,
                                                headers=self.headers,
                                                json=data)
            except Exception:
                time.sleep(1)
                continue
            requests_id = raw_response.headers['X-Request-Id']  # noqa
            self.release()

            if not stream:
                response = raw_response.json()

                if response is None:
                    print('Connection error, reconnect.')
                    # if connect error, frequent requests will casuse
                    # continuous unstable network, therefore wait here
                    # to slow down the request
                    self.wait()
                    continue
                if raw_response.status_code == 200:
                    msg = response['data']['choices'][0]['message']
                    return msg

                if (raw_response.status_code != 200):
                    if response['error']['code'] == 18:
                        # security issue
                        return 'error:unsafe'
                    if response['error']['code'] == 17:
                        return 'error:too long'
                    else:
                        print(raw_response.text)
                        from IPython import embed
                        embed()
                        exit()
                        time.sleep(1)
                        continue
            else:
                # stream data to msg
                raw_response.encoding = 'utf-8'

                if raw_response.status_code == 200:
                    response_text = raw_response.text
                    data_blocks = response_text.split('data:')
                    data_blocks = data_blocks[1:]

                    first_block = json.loads(data_blocks[0])
                    if first_block['status']['code'] != 0:
                        msg = f"error:{first_block['status']['code']},"
                        f" {first_block['status']['message']}"
                        self.logger.error(msg)
                        return msg

                    msg = ''
                    for i, part in enumerate(data_blocks):
                        # print(f'process {i}: {part}')
                        try:
                            if part.startswith('[DONE]'):
                                break

                            json_data = json.loads(part)
                            choices = json_data['data']['choices']
                            for c in choices:
                                delta = c.get('delta')
                                msg += delta
                        except json.decoder.JSONDecodeError as err:
                            print(err)
                            self.logger.error(f'Error decoding JSON: {part}')
                    return msg

                else:
                    print(raw_response.text,
                          raw_response.headers.get('X-Request-Id'))
                    time.sleep(1)
                    continue

        raise RuntimeError
