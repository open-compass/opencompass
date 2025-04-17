import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import httpx
import jieba
import requests
from tqdm import tqdm

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]
OPENAI_API_BASE = os.path.join(
    os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1/'),
    'chat/completions',
)
OPENAISDK_API_BASE = os.environ.get('OPENAI_BASE_URL',
                                    'https://api.openai.com/v1/')

O1_MODEL_LIST = ['o1', 'o3']


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
        openai_proxy_url (str, optional): An optional proxy url to use when
            connecting to OpenAI's API. When set to 'ENV', the url will be
            fetched from the environment variable $OPENAI_PROXY_URL.
            Defaults to None.
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

    def __init__(
        self,
        path: str = 'gpt-3.5-turbo',
        max_seq_len: int = 16384,
        query_per_second: int = 1,
        rpm_verbose: bool = False,
        retry: int = 2,
        key: Union[str, List[str]] = 'ENV',
        org: Optional[Union[str, List[str]]] = None,
        meta_template: Optional[Dict] = None,
        openai_api_base: str = OPENAI_API_BASE,
        openai_proxy_url: Optional[str] = None,
        mode: str = 'none',
        logprobs: Optional[bool] = False,
        top_logprobs: Optional[int] = None,
        temperature: Optional[float] = None,
        tokenizer_path: Optional[str] = None,
        extra_body: Optional[Dict] = None,
        verbose: bool = False,
    ):

        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            query_per_second=query_per_second,
            rpm_verbose=rpm_verbose,
            retry=retry,
            verbose=verbose,
        )
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

        if openai_proxy_url == 'ENV':
            if 'OPENAI_PROXY_URL' not in os.environ:
                raise ValueError('OPENAI_PROXY_URL is not set.')
            self.proxy_url = os.getenv('OPENAI_PROXY_URL')
        else:
            self.proxy_url = openai_proxy_url

        self.path = path

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
        temperature: float = 0.7,
        **kwargs,
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

        Returns:
            List[str]: A list of generated strings.
        """
        if self.temperature is not None:
            temperature = self.temperature

        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(
                        self._generate,
                        inputs,
                        [max_out_len] * len(inputs),
                        [temperature] * len(inputs),
                    ),
                    total=len(inputs),
                    desc='Inferencing',
                ))
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

        messages, max_out_len = self._preprocess_messages(
            input, max_out_len, self.max_seq_len, self.mode,
            self.get_token_len)

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
                if any(model in self.path for model in O1_MODEL_LIST):
                    self.logger.warning(
                        f"'max_token' is unsupported for model {self.path}")
                    self.logger.warning(
                        f'We use max_out_len: {max_out_len} for this query')
                    data = dict(
                        model=self.path,
                        messages=messages,
                        max_completion_tokens=max_out_len,
                        n=1,
                        logprobs=self.logprobs,
                        top_logprobs=self.top_logprobs,
                        stop=None,
                        temperature=temperature,
                    )
                else:
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

                if self.proxy_url is None:
                    raw_response = requests.post(url,
                                                 headers=header,
                                                 data=json.dumps(data))
                else:
                    proxies = {
                        'http': self.proxy_url,
                        'https': self.proxy_url,
                    }
                    if self.verbose:
                        self.logger.debug(
                            f'Start send query to {self.proxy_url}')
                    raw_response = requests.post(
                        url,
                        headers=header,
                        data=json.dumps(data),
                        proxies=proxies,
                    )

                    if self.verbose:
                        self.logger.debug(
                            f'Get response from {self.proxy_url}')

            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            try:
                if raw_response.status_code != 200:
                    self.logger.error(f'Request failed with status code '
                                      f'{raw_response.status_code}, response: '
                                      f'{raw_response.content.decode()}')
                    continue
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error(f'JsonDecode error, got status code '
                                  f'{raw_response.status_code}, response: '
                                  f'{raw_response.content.decode()}')
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

                    self.logger.error(
                        'Find error message in response: ',
                        str(response['error']),
                    )
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
        assert self.tokenizer_path or self.path
        try:
            if self.verbose:
                self.logger.info(f'Used tokenizer_path: {self.tokenizer_path}')
            tokenizer_path = (self.tokenizer_path
                              if self.tokenizer_path else self.path)
            try:
                if self.verbose:
                    self.logger.info(
                        f'Start load tiktoken encoding: {tokenizer_path}')
                enc = self.tiktoken.encoding_for_model(tokenizer_path)
                if self.verbose:
                    self.logger.info(
                        f'Successfully tiktoken encoding: {tokenizer_path}')
                return len(enc.encode(prompt, disallowed_special=()))
            except Exception as e:
                self.logger.warn(f'{e}, tiktoken encoding cannot load '
                                 f'{tokenizer_path}')
                from transformers import AutoTokenizer

                if self.hf_tokenizer is None:
                    if self.verbose:
                        self.logger.info(
                            f'Start load hf tokenizer: {tokenizer_path}')
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path, trust_remote_code=True)
                    self.logger.info(
                        f'Successfully load HF Tokenizer from {tokenizer_path}'
                    )
                return len(self.hf_tokenizer(prompt).input_ids)
        except Exception:
            self.logger.warn(
                'Can not get tokenizer automatically, '
                'will use default tokenizer gpt-4 for length calculation.')
            default_tokenizer = 'gpt-4'

            enc = self.tiktoken.encoding_for_model(default_tokenizer)
            if self.verbose:
                self.logger.info(
                    f'Successfully load default tiktoken tokenizer: '
                    f' {default_tokenizer}')
            return len(enc.encode(prompt, disallowed_special=()))

    def _bin_trim(self, prompt: str, num_token: int, mode: str) -> str:
        """Get a suffix of prompt which is no longer than num_token tokens.

        Args:
            prompt (str): Input string.
            num_token (int): The upper bound of token numbers.
            mode (str): The method of input truncation
            ('front', 'mid', or 'rear')

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
            if mode == 'front':
                cur_prompt = sep.join(words[-mid:])
            elif mode == 'mid':
                cur_prompt = sep.join(words[:mid]) + sep.join(words[-mid:])
            elif mode == 'rear':
                cur_prompt = sep.join(words[:mid])

            if self.get_token_len(cur_prompt) <= num_token:
                l = mid  # noqa: E741
            else:
                r = mid

        if mode == 'front':
            prompt = sep.join(words[-l:])
        elif mode == 'mid':
            prompt = sep.join(words[:l]) + sep.join(words[-l:])
        elif mode == 'rear':
            prompt = sep.join(words[:l])
        return prompt

    def _preprocess_messages(
        self,
        input: Union[str, PromptList],
        max_out_len: int,
        max_seq_len: int,
        mode: str,
        get_token_len_func,
    ) -> tuple[List[Dict], int]:
        """Preprocess input into messages format and calculate max output
        length.

        Args:
            input: Input prompt as string or PromptList
            max_out_len: Maximum output length
            max_seq_len: Maximum sequence length
            mode: The method of input truncation
            get_token_len_func: Function to calculate token length

        Returns:
            tuple: (processed messages list, adjusted max_out_len)
        """
        # Check input length when mode is 'none'
        if mode == 'none':
            input_len = (get_token_len_func(input) if isinstance(
                input, str) else sum(
                    get_token_len_func(item['prompt']) for item in input))
            if input_len > max_seq_len:
                raise ValueError(
                    f'Input length ({input_len}) exceeds max_seq_len '
                    f'({max_seq_len}) and mode is set to "none". Please '
                    f'either change the mode or increase the max_seq_len.')

        # Trim input if needed
        def bin_trim_wrapper(text):
            trim_length = max_seq_len - 100
            if max_out_len is not None:
                trim_length -= max_out_len
            return self._bin_trim(text, trim_length, mode)

        if isinstance(input, str) and mode != 'none':
            input = bin_trim_wrapper(input)
        # Convert input to messages format
        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
            input_len = get_token_len_func(input)
        else:
            messages = []
            processed_prompts = []
            for item in input:
                input_content = item['prompt']
                if mode != 'none':
                    input_content = bin_trim_wrapper(input_content)
                processed_prompts.append(input_content)
                msg = {'content': input_content}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)
            input_len = sum(
                get_token_len_func(prompt) for prompt in processed_prompts)

        # Adjust max_out_len
        if max_out_len is not None:
            original_max_out_len = max_out_len
            max_out_len = min(max_out_len, max_seq_len - input_len - 100)
            if max_out_len <= 0:
                raise ValueError(
                    f'max_out_len ({max_out_len}) is less than or equal to 0. '
                    f'This may be due to input length ({input_len}) being too '
                    f'close to max_seq_len ({max_seq_len}). Please increase '
                    f'max_seq_len or use a truncation mode other than "none".')
            if max_out_len < original_max_out_len:
                self.logger.warning(
                    f'max_out_len was truncated from {original_max_out_len} '
                    f'to {max_out_len} due to input length')

        return messages, max_out_len


class OpenAISDK(OpenAI):

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 16384,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 key: str | List[str] = 'ENV',
                 org: str | List[str] | None = None,
                 meta_template: Dict | None = None,
                 openai_api_base: str | List[str] = OPENAISDK_API_BASE,
                 openai_proxy_url: Optional[str] = None,
                 mode: str = 'none',
                 logprobs: bool | None = False,
                 top_logprobs: int | None = None,
                 temperature: float | None = None,
                 tokenizer_path: str | None = None,
                 extra_body: Dict | None = None,
                 verbose: bool = False,
                 status_code_mappings: dict = {},
                 think_tag: str = '</think>'):
        super().__init__(
            path,
            max_seq_len,
            query_per_second,
            rpm_verbose,
            retry,
            key,
            org,
            meta_template,
            openai_api_base,
            openai_proxy_url,
            mode,
            logprobs,
            top_logprobs,
            temperature,
            tokenizer_path,
            extra_body,
            verbose=verbose,
        )
        from openai import OpenAI

        # support multiple api_base for acceleration
        if isinstance(openai_api_base, List):
            self.openai_api_base = random.choice(openai_api_base)
        else:
            self.openai_api_base = openai_api_base

        if self.proxy_url is None:
            self.openai_client = OpenAI(base_url=self.openai_api_base,
                                        api_key=key)
        else:
            proxies = {
                'http://': self.proxy_url,
                'https://': self.proxy_url,
            }

            self.openai_client = OpenAI(
                base_url=self.openai_api_base,
                api_key=key,
                http_client=httpx.Client(proxies=proxies),
            )
        if self.verbose:
            self.logger.info(f'Used openai_client: {self.openai_client}')
        self.status_code_mappings = status_code_mappings
        self.think_tag = think_tag

    def _generate(self,
                  input: PromptList | str,
                  max_out_len: int,
                  temperature: float,
                  timeout: int = 3600) -> str:
        """Generate results given a list of inputs.

        Args:
            input (PromptType): A string or PromptDict.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use.
            timeout (int, optional): Timeout in seconds for the API call.
                Defaults to 3600 (60 minutes).

        Returns:
            str: The generated string.
        """
        from openai import APIStatusError, BadRequestError

        assert isinstance(input, (str, PromptList))

        messages, max_out_len = self._preprocess_messages(
            input, max_out_len, self.max_seq_len, self.mode,
            self.get_token_len)

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            if any(model in self.path for model in O1_MODEL_LIST):
                self.logger.warning(
                    f"'max_token' is unsupported for model {self.path}")
                self.logger.warning(
                    f'We use max_out_len: {max_out_len} for this query')
                query_data = dict(
                    model=self.path,
                    max_completion_tokens=max_out_len,
                    n=1,
                    messages=messages,
                    extra_body=self.extra_body,
                )
            else:
                query_data = dict(
                    model=self.path,
                    max_tokens=max_out_len,
                    n=1,
                    temperature=self.temperature,
                    messages=messages,
                    extra_body=self.extra_body,
                )

            try:
                if self.verbose:
                    self.logger.info('Start calling OpenAI API')
                responses = self.openai_client.chat.completions.create(
                    **query_data, timeout=timeout)  # timeout in seconds
                if self.verbose:
                    self.logger.info(
                        'Successfully get response from OpenAI API')
                    try:
                        self.logger.info(responses)
                    except Exception:
                        pass  # noqa F841

                # Check if response is empty or content is empty
                if not responses.choices or not responses.choices[
                        0].message.content:
                    self.logger.error(
                        'API response is empty, it might be due to excessive '
                        'input length or an internal server error '
                        'from your API provider.')
                    num_retries += 1
                    # Continue to retry instead of returning empty response
                    continue
                # If the model has reasoning_content, concat it
                # with the content
                if hasattr(responses.choices[0].message, 'reasoning_content'):
                    return (responses.choices[0].message.reasoning_content +
                            self.think_tag +
                            responses.choices[0].message.content)

                return responses.choices[0].message.content

            except (BadRequestError, APIStatusError) as e:
                # Handle BadRequest status
                # You can specify self.status_code_mappings to bypass \
                # API sensitivity blocks
                # For example: status_code_mappings={400: 'Input data \
                # may contain inappropriate content.'}
                status_code = e.status_code
                if (status_code is not None
                        and status_code in self.status_code_mappings):
                    error_message = self.status_code_mappings[status_code]
                    self.logger.error(
                        f'error occurs at {self.openai_api_base}')
                    self.logger.info(f'Status Code: {status_code}, \n'
                                     f'Original Error Message: {e}, \n'
                                     f'Return Message: {error_message} ')
                    return error_message
                else:
                    self.logger.error(
                        f'error occurs at {self.openai_api_base}')
                    self.logger.error(e)
            except Exception as e:
                self.logger.error(f'error occurs at {self.openai_api_base}')
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError('Calling OpenAI API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')
