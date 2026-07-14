import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import numpy as np
import requests

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


def valid_str(string, coding='utf-8'):
    """Decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


class TurboMindAPIModel(BaseModel):
    """Model wrapper for lmdeploy api server.

    Args:
        api_addr (str): The address (ip:port format) of lmdeploy's
            api server.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        end_str (str, optional): Whether to trim generated strings with end_str
            if the model has special ending strings that are not handled well.
            Defaults to None.
        temperature (float, optional): What sampling temperature to use. If
            set, overrides the temperature passed to ``generate``.
        top_p (float, optional): Nucleus sampling parameter passed to the
            lmdeploy server. Defaults to 0.8, matching the historical wrapper
            behavior.
        top_k (int, optional): Top-k sampling parameter passed to the lmdeploy
            server. Defaults to 1 for deterministic evaluation.
        gen_config (Dict, optional): Extra generation parameters passed to
            lmdeploy's completion API, such as ``random_seed``. Values in
            ``gen_config`` override ``top_p`` and ``top_k`` defaults.
    """

    is_api: bool = True

    def __init__(self,
                 model_name: str = None,
                 api_addr: str = 'http://0.0.0.0:23333',
                 api_key: str | None = None,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 end_str: Optional[str] = None,
                 temperature: float = None,
                 top_p: Optional[float] = 0.8,
                 top_k: Optional[int] = 1,
                 gen_config: Optional[Dict] = None,
                 **kwargs):
        super().__init__(path='',
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)
        from lmdeploy.serve.openai.api_client import APIClient
        self.chatbot = APIClient(api_addr, api_key)
        self.model_name = model_name
        self.logger = get_logger()
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']
        self.api_addr = api_addr
        self.encode_addr = f'{api_addr.rstrip("/")}/v1/encode'
        self.ppl_addr = f'{api_addr.rstrip("/")}/get_ppl'
        self.end_str = end_str
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.gen_config = dict(gen_config or {})

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
        temperature: float = 1.0,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic. Defaults to 0.7.
            end_str (str, optional): Whether to trim generated strings
                with end_str if the model has special ending strings
                that are not handled well.
                Defaults to None.
        Returns:
            List[str]: A list of generated strings.
        """

        if self.temperature is not None:
            temperature = self.temperature

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs),
                             [self.end_str] * len(inputs)))
        return results

    def _encode(self, text):
        """Encode text or list of texts to token ids via /v1/encode.

        Bypasses lmdeploy's APIClient.encode to include ``model`` field
        in the request body, which is required by some API proxies.
        """
        if isinstance(text, list):
            all_ids, all_lens = [], []
            for t in text:
                ids, lens = self._encode(t)
                all_ids.append(ids)
                all_lens.append(lens)
            return all_ids, all_lens
        resp = requests.post(self.encode_addr,
                             headers=self.chatbot.headers,
                             json={
                                 'input': text,
                                 'do_preprocess': False,
                                 'add_bos': True,
                                 'model': self.model_name
                             })
        output = resp.json()
        return output['input_ids'], output['length']

    def get_token_len(self, prompt: str) -> int:
        _, length = self._encode(prompt)
        return length

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

    def _get_generation_kwargs(self, max_out_len: int,
                               temperature: float) -> tuple[int, Dict]:
        gen_config = copy.deepcopy(self.gen_config)
        gen_config.setdefault('temperature', temperature)
        if self.top_p is not None:
            gen_config.setdefault('top_p', self.top_p)
        if self.top_k is not None:
            gen_config.setdefault('top_k', self.top_k)
        gen_config.setdefault('session_id', threading.current_thread().ident)
        max_tokens = gen_config.pop(
            'max_tokens', gen_config.pop('max_new_tokens', max_out_len))
        return max_tokens, gen_config

    def _generate(self, prompt: PromptType, max_out_len: int,
                  temperature: float, end_str: str) -> str:
        """Generate results given a list of inputs.

        Args:
            prompt (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.

        Returns:
            str: The generated string.
        """
        assert type(
            prompt) is str, 'We only support string for TurboMind RPC API'

        max_tokens, gen_kwargs = self._get_generation_kwargs(
            max_out_len, temperature)
        response = ''
        for output in self.chatbot.completions_v1(
                prompt=prompt,
                model=self.model_name,
                max_tokens=max_tokens,
                **gen_kwargs,
        ):
            response += output['choices'][0]['text']
        response = valid_str(response)
        if end_str:
            response = response.split(end_str)[0]
        return response

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> np.ndarray:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): Reserved for compatibility.
                lmdeploy serving's /get_ppl endpoint scores the full input.

        Returns:
            np.ndarray: The perplexity scores in shape of (N,).
        """
        assert isinstance(
            inputs, List), f'List(str) is expected, but got {type(inputs)}'

        if len(inputs) == 0:
            return np.array([])

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._get_ppl, inputs))
        return np.array(results)

    def _get_ppl(self, prompt: Union[str, List[int]]) -> float:
        assert type(prompt) is str or (
            type(prompt) is list
            and all(type(token_id) is int for token_id in prompt)
        ), 'We only support string or token ids for TurboMind RPC API'

        raw_response = requests.post(self.ppl_addr,
                                     headers=self.chatbot.headers,
                                     json={
                                         'input': prompt,
                                         'model': self.model_name
                                     },
                                     stream=False)

        if not raw_response.ok:
            raise RuntimeError('Calling TurboMindAPIModel /get_ppl failed '
                               f'for {self.ppl_addr}: status code '
                               f'{raw_response.status_code}, response '
                               f'{raw_response.text}')

        response = raw_response.json()
        return float(response['ppl'])

    def get_loglikelihood(
            self,
            inputs: List[str],
            conts: List[str],
            mask_length: Optional[List[int]] = None) -> np.ndarray:
        """Get loglikelihood scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            conts (List[str]): A list of continuation strings.
            mask_length (Optional[List[int]]): Reserved for compatibility.

        Returns:
            np.ndarray: The loglikelihood scores in shape of (N,).
        """
        assert isinstance(
            inputs, List), f'List(str) is expected, but got {type(inputs)}'
        assert isinstance(
            conts, List), f'List(str) is expected, but got {type(conts)}'
        assert len(inputs) == len(conts), \
            'inputs and conts must have the same length'

        if len(inputs) == 0:
            return np.array([])

        context_inputs = [
            text.replace(cont, '') for text, cont in zip(inputs, conts)
        ]
        input_ids, input_lengths = self._encode(inputs)
        context_ids, context_lengths = self._encode(context_inputs)

        with ThreadPoolExecutor() as executor:
            ppl_results = list(
                executor.map(self._get_ppl, input_ids + context_ids))

        batch_size = len(inputs)
        full_ppls = ppl_results[:batch_size]
        context_ppls = ppl_results[batch_size:]
        results = []
        for full_ppl, full_len, context_ppl, context_len in zip(
                full_ppls, input_lengths, context_ppls, context_lengths):
            logit_sum = full_ppl * full_len
            logit_part = context_ppl * context_len
            results.append(-(logit_sum - logit_part))
        return np.array(results)
