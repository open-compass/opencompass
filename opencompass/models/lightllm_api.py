from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import requests
import json

from opencompass.registry import MODELS
from .base_api import BaseAPIModel
from opencompass.utils.logging import get_logger


@MODELS.register_module()
class LightllmApi(BaseAPIModel):

    is_api: bool = True

    def __init__(
        self,
        path: str = 'LightllmApi',
        url: str = 'http://localhost:8080/generate',
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        generation_kwargs: Optional[Dict] = None,
    ):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         retry=retry)
        self.logger = get_logger()
        self.url = url
        if generation_kwargs is not None:
            self.generation_kwargs = generation_kwargs
        else:
            self.generation_kwargs = {}
        self.do_sample = self.generation_kwargs.get('do_sample', False)
        self.ignore_eos = self.generation_kwargs.get('ignore_eos', False)

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
                             [max_out_len] * len(inputs)))
        return results

    def _generate(self, input: str, max_out_len: int) -> str:
        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()
            header = {'content-type': 'application/json'}
            try:
                data = dict(
                    inputs=input,
                    parameters=dict(
                        do_sample=self.do_sample,
                        ignore_eos=self.ignore_eos,
                        max_new_tokens=max_out_len
                    )
                )
                raw_response = requests.post(self.url,
                                             headers=header,
                                             data=json.dumps(data))
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
                return response['generated_text']
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
            max_num_retries += 1

        raise RuntimeError('Calling LightllmApi failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')
