import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import jieba
import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]
OPENAI_API_BASE = 'https://api.openai.com/v1/chat/completions'


@MODELS.register_module()
class OpenAI(BaseAPIModel):
    """Model wrapper around OpenAI's models.

    Args:
        path (str): The name of OpenAI's model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        key (str or List[str]): OpenAI key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
            list, the keys will be used in round-robin manner. Defaults to
            'ENV'.
        org (str or List[str], optional): OpenAI organization(s). If not
            specified, OpenAI uses the default organization bound to each API
            key. If specified, the orgs will be posted with each request in
            round-robin manner. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        openai_api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'front','mid' and 'rear' represents the part
            of input to truncate. Defaults to 'none'.
        temperature (float, optional): What sampling temperature to use.
            If not None, will override the temperature in the `generate()`
            call. Defaults to None.
        tokenizer_path (str, optional): The path to the tokenizer. Use path if
            'tokenizer_path' is None, otherwise use the 'tokenizer_path'.
            Defaults to None.
        extra_body (Dict, optional): Add additional JSON properties to
            the request
    """

    is_api: bool = True

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 key: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = None,
                 openai_api_base: str = OPENAI_API_BASE,
                 mode: str = 'none',
                 logprobs: Optional[bool] = False,
                 top_logprobs: Optional[int] = None,
                 temperature: Optional[float] = None,
                 tokenizer_path: Optional[str] = None,
                 extra_body: Optional[Dict] = None):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry)
        import tiktoken
        self.tiktoken = tiktoken
        self.temperature = temperature
        assert mode in ['none', 'front', 'mid', 'rear']
        self.mode = mode
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.tokenizer_path = tokenizer_path
        self.hf_tokenizer = None
        self.extra_body = extra_body

        if isinstance(key, str):
            if key == 'ENV':
                if 'OPENAI_API_KEY' not in os.environ:
                    raise ValueError('OpenAI API key is not set.')
                self.keys = os.getenv('OPENAI_API_KEY').split(',')
            else:
                self.keys = [key]
        else:
            self.keys = key

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0
        self.url = openai_api_base
        self.path = path

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512,
                 temperature: float = 0.7,
                 **kwargs) -> List[str]:
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

        Returns:
            List[str]: A list of generated strings.
        """
        if self.temperature is not None:
            temperature = self.temperature

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results

    def _generate(self, input: PromptType, max_out_len: int,
                  temperature: float) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (PromptType): A string or PromptDict.
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
        assert isinstance(input, (str, PromptList))

        # max num token for gpt-3.5-turbo is 4097
        context_window = 4096
        if '32k' in self.path:
            context_window = 32768
        elif '16k' in self.path:
            context_window = 16384
        elif 'gpt-4' in self.path:
            context_window = 8192

        # will leave 100 tokens as prompt buffer, triggered if input is str
        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - 100 - max_out_len)

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
                    msg['role'] = 'system'
                messages.append(msg)

        # Hold out 100 tokens due to potential errors in tiktoken calculation
        try:
            max_out_len = min(
                max_out_len,
                context_window - self.get_token_len(str(input)) - 100)
        except KeyError:
            max_out_len = max_out_len
        if max_out_len <= 0:
            return ''

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()

            with Lock():
                if len(self.invalid_keys) == len(self.keys):
                    raise RuntimeError('All keys have insufficient quota.')

                # find the next valid key
                while True:
                    self.key_ctr += 1
                    if self.key_ctr == len(self.keys):
                        self.key_ctr = 0

                    if self.keys[self.key_ctr] not in self.invalid_keys:
                        break

                key = self.keys[self.key_ctr]

            header = {
                'Authorization': f'Bearer {key}',
                'content-type': 'application/json',
                'api-key': key,
            }

            if self.orgs:
                with Lock():
                    self.org_ctr += 1
                    if self.org_ctr == len(self.orgs):
                        self.org_ctr = 0
                header['OpenAI-Organization'] = self.orgs[self.org_ctr]

            try:
                data = dict(
                    model=self.path,
                    messages=messages,
                    max_tokens=max_out_len,
                    n=1,
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                    stop=None,
                    temperature=temperature,
                )
                if self.extra_body:
                    data.update(self.extra_body)
                if isinstance(self.url, list):
                    import random
                    url = self.url[random.randint(0, len(self.url) - 1)]
                else:
                    url = self.url
                raw_response = requests.post(url,
                                             headers=header,
                                             data=json.dumps(data))
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
                continue
            self.logger.debug(str(response))
            try:
                if self.logprobs:
                    return response['choices']
                else:
                    return response['choices'][0]['message']['content'].strip()
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(10)
                        self.logger.warn('Rate limit exceeded, retrying...')
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        self.invalid_keys.add(key)
                        self.logger.warn(f'insufficient_quota key: {key}')
                        continue
                    elif response['error']['code'] == 'invalid_prompt':
                        self.logger.warn('Invalid prompt:', str(input))
                        return ''
                    elif response['error']['type'] == 'invalid_prompt':
                        self.logger.warn('Invalid prompt:', str(input))
                        return ''

                    self.logger.error('Find error message in response: ',
                                      str(response['error']))
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        if self.tokenizer_path:
            try:
                enc = self.tiktoken.encoding_for_model(self.tokenizer_path)
                return len(enc.encode(prompt))
            except Exception as e:
                self.logger.warn(f'{e}, tiktoken encoding cannot load '
                                 '{self.tokenizer_path}')
                from transformers import AutoTokenizer
                if self.hf_tokenizer is None:
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(
                        self.tokenizer_path, trust_remote_code=True)
                    self.logger.infer(
                        f'Tokenizer is loaded from {self.tokenizer_path}')
                return len(self.hf_tokenizer(prompt).input_ids)
        else:
            enc = self.tiktoken.encoding_for_model(self.path)
            return len(enc.encode(prompt))

    def bin_trim(self, prompt: str, num_token: int) -> str:
        """Get a suffix of prompt which is no longer than num_token tokens.

        Args:
            prompt (str): Input string.
            num_token (int): The upper bound of token numbers.

        Returns:
            str: The trimmed prompt.
        """
        token_len = self.get_token_len(prompt)
        if token_len <= num_token:
            return prompt
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if pattern.search(prompt):
            words = list(jieba.cut(prompt, cut_all=False))
            sep = ''
        else:
            words = prompt.split(' ')
            sep = ' '

        l, r = 1, len(words)
        while l + 2 < r:
            mid = (l + r) // 2
            if self.mode == 'front':
                cur_prompt = sep.join(words[-mid:])
            elif self.mode == 'mid':
                cur_prompt = sep.join(words[:mid]) + sep.join(words[-mid:])
            elif self.mode == 'rear':
                cur_prompt = sep.join(words[:mid])

            if self.get_token_len(cur_prompt) <= num_token:
                l = mid  # noqa: E741
            else:
                r = mid

        if self.mode == 'front':
            prompt = sep.join(words[-l:])
        elif self.mode == 'mid':
            prompt = sep.join(words[:l]) + sep.join(words[-l:])
        elif self.mode == 'rear':
            prompt = sep.join(words[:l])
        return prompt


class OpenAISDK(OpenAI):

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 key: str | List[str] = 'ENV',
                 org: str | List[str] | None = None,
                 meta_template: Dict | None = None,
                 openai_api_base: str = OPENAI_API_BASE,
                 mode: str = 'none',
                 logprobs: bool | None = False,
                 top_logprobs: int | None = None,
                 temperature: float | None = None,
                 tokenizer_path: str | None = None,
                 extra_body: Dict | None = None):
        super().__init__(path, max_seq_len, query_per_second, rpm_verbose,
                         retry, key, org, meta_template, openai_api_base, mode,
                         logprobs, top_logprobs, temperature, tokenizer_path,
                         extra_body)
        from openai import OpenAI
        self.opeanai_cleint = OpenAI(base_url=openai_api_base, api_key=key)

    def _generate(self, input: PromptList | str, max_out_len: int,
                  temperature: float) -> str:
        assert isinstance(input, (str, PromptList))

        # max num token for gpt-3.5-turbo is 4097
        context_window = 4096
        if '32k' in self.path:
            context_window = 32768
        elif '16k' in self.path:
            context_window = 16384
        elif 'gpt-4' in self.path:
            context_window = 8192

        # will leave 100 tokens as prompt buffer, triggered if input is str
        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - 100 - max_out_len)

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
                    msg['role'] = 'system'
                messages.append(msg)

        # Hold out 100 tokens due to potential errors in tiktoken calculation
        # try:
        #     max_out_len = min(
        #         max_out_len,
        #         context_window - self.get_token_len(str(input)) - 100)
        # except KeyError:
        #     max_out_len = max_out_len
        # if max_out_len <= 0:
        #     return ''

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                responses = self.opeanai_cleint.chat.completions.create(
                    model=self.path,
                    max_tokens=max_out_len,
                    n=1,
                    temperature=self.temperature,
                    messages=messages)
                return responses.choices[0].message.content
            except Exception as e:
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError('Calling OpenAI API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')
