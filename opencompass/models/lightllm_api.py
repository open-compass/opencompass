import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import requests

from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger

from .base_api import BaseAPIModel


@MODELS.register_module()
class LightllmAPI(BaseAPIModel):

    is_api: bool = True

    def __init__(
            self,
            path: str = 'LightllmAPI',
            url: str = 'http://localhost:8080/generate',
            max_seq_len: int = 2048,
            meta_template: Optional[Dict] = None,
            retry: int = 2,
            generation_kwargs: Optional[Dict] = dict(),
    ):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         retry=retry,
                         generation_kwargs=generation_kwargs)
        self.logger = get_logger()
        self.url = url
        self.generation_kwargs = generation_kwargs
        self.max_out_len = self.generation_kwargs.get('max_new_tokens', 1024)

    def generate(self, inputs: List[str], max_out_len: int,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [self.max_out_len] * len(inputs)))
        return results

    def _generate(self, input: str, max_out_len: int) -> str:
        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()
            header = {'content-type': 'application/json'}
            try:
                data = dict(inputs=input, parameters=self.generation_kwargs)
                raw_response = requests.post(self.url,
                                             headers=header,
                                             data=json.dumps(data))
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
                generated_text = response['generated_text']
                if isinstance(generated_text, list):
                    generated_text = generated_text[0]
                return generated_text
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
            max_num_retries += 1

        raise RuntimeError('Calling LightllmAPI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def get_ppl(self, inputs: List[str], max_out_len: int,
                **kwargs) -> List[float]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._get_ppl, inputs,
                             [self.max_out_len] * len(inputs)))
        return np.array(results)

    def _get_ppl(self, input: str, max_out_len: int) -> float:
        max_num_retries = 0
        if max_out_len is None:
            max_out_len = 1
        while max_num_retries < self.retry:
            self.wait()
            header = {'content-type': 'application/json'}
            try:
                data = dict(inputs=input, parameters=self.generation_kwargs)
                raw_response = requests.post(self.url,
                                             headers=header,
                                             data=json.dumps(data))
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()

                assert ('prompt_token_ids' in response and 'prompt_logprobs'
                        in response), 'prompt_token_ids and prompt_logprobs \
                    must be in the output. \
                    Please consider adding \
                    --return_all_prompt_logprobs argument \
                    when starting your lightllm service.'

                prompt_token_ids = response['prompt_token_ids'][1:]
                prompt_logprobs = [
                    item[1] for item in response['prompt_logprobs']
                ]
                logprobs = [
                    item[str(token_id)] for token_id, item in zip(
                        prompt_token_ids, prompt_logprobs)
                ]
                if len(logprobs) == 0:
                    return 0.0
                ce_loss = -sum(logprobs) / len(logprobs)
                return ce_loss
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
            max_num_retries += 1
        raise RuntimeError('Calling LightllmAPI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')
