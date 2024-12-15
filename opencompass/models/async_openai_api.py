import contextlib
import multiprocessing
import os
import re
from typing import Dict, List, Optional, Union

import jieba
import weakref
from typing import Literal, Tuple, Iterable

from opencompass.utils.prompt import PromptList
from opencompass.models.base_api import AsyncTokenBucket, BaseAPIModel

import threading
import asyncio
from typing import cast
from contextlib import contextmanager


PromptType = Union[PromptList, str]
OPENAI_API_BASE = os.path.join(
    os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1/'),
    'chat/completions')


class _APIModelState:
    _instance: Dict[str, weakref.ReferenceType["_APIModelState"]] = {}
    _count: int
    _concurrency: int
    _locks = [threading.Lock(), multiprocessing.Lock()]

    def __init__(self, *, name: str, concurrency: int, query_per_second=1) -> None:
        self._name = name
        self._count = 0
        self._concurrency = concurrency
        self._token_bucket = AsyncTokenBucket(rate=query_per_second)

        self._count += 1
        self._concurrency = max(1, self._concurrency // self._count)

    @property
    def concurrency(self) -> int:
        # If update and concurrency are called simultaneously, the values
        # returned here may be inaccurate, but the impact is likely minimal
        return self._concurrency

    async def acquire(self):
        return await self._token_bucket.acquire()

    @property
    def rpm(self):
        return self._token_bucket.rpm

    @property
    def name(self) -> str:
        return self._name

    @property
    def count(self):
        return self._count

    @classmethod
    def _cleanup(cls, ref: weakref.ReferenceType["_APIModelState"]):
        with cls._lock():
            self: _APIModelState = ref()  # type: ignore
            cls._instance.pop(self._name)

    def __new__(cls, name: str, *args, **kwargs) -> "_APIModelState":
        with cls._lock():
            if name not in cls._instance:
                self = super().__new__(cls)
                cls._instance[name] = weakref.ref(self, cls._cleanup)
            return cls._instance[name]()  # type: ignore

    @classmethod
    @contextmanager
    def _lock(cls):
        with contextlib.ExitStack() as stack:
            [stack.enter_context(lock) for lock in cls._locks]
            yield



class AsyncOpenAISDK(BaseAPIModel):
    states: Dict[str, _APIModelState] = {}

    def __init__(
        self,
        path: str = 'gpt-3.5-turbo',
        max_seq_len: int | None = None,  # type: ignore
        query_per_second: int = 1,
        retry: int = 2,
        key: str = 'ENV',
        org: str | List[str] | None = None,
        meta_template: Dict | None = None,
        openai_api_base: str = OPENAI_API_BASE,
        openai_proxy_url: Optional[str] = None,
        mode: Literal['none', 'front', 'mid', 'rear'] = 'none',
        logprobs: bool | None = False,
        top_logprobs: int | None = None,
        temperature: float | None = None,
        tokenizer_path: str | None = None,
        extra_body: Dict | None = None,
        max_completion_tokens: int = 16384,
        verbose: bool = False,
        concurrency: int = 64,
        status_code_mappings: dict = {},
    ):
        from openai import AsyncOpenAI

        assert mode in ['none', 'front', 'mid', 'rear']
        self.mode = mode
        state_key = self._get_state_key(api_base=openai_api_base, model_name=path)
        if state_key not in AsyncOpenAISDK.states:
            AsyncOpenAISDK.states[path] = _APIModelState(
                name=state_key,
                concurrency=concurrency,
                query_per_second=query_per_second,
            )
        self.state = AsyncOpenAISDK.states[path]
        self.openai_client = AsyncOpenAI(base_url=openai_api_base, api_key=key)

        if max_seq_len is None:
            if '16k' in path:
                max_seq_len = 16384
            elif 'gpt-4' in path:
                max_seq_len = 8192
            elif 'gpt-3.5' in path:
                max_seq_len = 4097
            else:
                max_seq_len = 32768
        else:
            max_seq_len = max_seq_len

        super().__init__(path=path, max_seq_len=max_seq_len, meta_template=meta_template, retry=retry)

        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.tokenizer_path = tokenizer_path
        self.hf_tokenizer = None
        self.extra_body = extra_body
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.openai_api_base = openai_api_base
        self.concurrency = concurrency

        self.status_code_mappings = status_code_mappings

        if openai_proxy_url == 'ENV':
            if 'OPENAI_PROXY_URL' not in os.environ:
                raise ValueError('OPENAI_PROXY_URL is not set.')
            self.proxy_url = os.getenv('OPENAI_PROXY_URL')
        else:
            self.proxy_url = openai_proxy_url

    async def generate(self,  # type: ignore
                 inputs: Iterable[PromptType],
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

        # TODO: This should be an AsyncGenerator if an real `AsyncInference` has been implemented
        tasks_queue: List[asyncio.Future] = []
        results_queue: List[Tuple[int, str]] = []
        inputs_iter = enumerate(inputs)

        data_stop = False
        while not (data_stop and not tasks_queue):
            concurrency = self.state.concurrency

            if tasks_queue:
                done, pending = await asyncio.wait(tasks_queue, return_when=asyncio.FIRST_COMPLETED)
                tasks_queue = list(pending)
                for queue in done:
                    result: Tuple[int, str] = queue.result()
                    results_queue.append(result)

            while not data_stop and len(tasks_queue) < concurrency:
                try:
                    index, _input = next(inputs_iter)
                except StopIteration:
                    data_stop = True
                    break
                tasks_queue.append(
                    asyncio.create_task(
                        self._generate(
                            input=_input,
                            max_out_len=self.max_completion_tokens or max_out_len,
                            temperature=temperature,
                            index=index,
                        )
                    )
                )
        results_queue.sort()
        return [item[1] for item in results_queue]

    async def generate_from_template(self, templates: List[PromptType],  # type: ignore
                               max_out_len: int, **kwargs):
        """Generate completion from a list of templates.

        Args:
            templates (List[PromptType]): A list of templates.
            max_out_len (int): The maximum length of the output.
        """
        inputs = self.parse_template(templates, mode='gen')  # type: ignore
        return await self.generate(inputs, max_out_len=max_out_len, **kwargs)

    async def _generate(self, input: PromptList | str, max_out_len: int,
                  temperature: float, index: int) -> Tuple[int, str]:
        from openai import APIStatusError, BadRequestError
        assert isinstance(input, (str, PromptList))

        # max num token for gpt-3.5-turbo is 4097
        # Most models' token limits are above 32k

        # will leave 100 tokens as prompt buffer, triggered if input is str
        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(
                input,
                context_window - 100 - max_out_len,
                cast(Literal['front', 'mid', 'rear'], self.mode),
            )

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
            await self.state.acquire()

            query_data = dict(
                model=self.path,
                max_tokens=max_out_len,
                n=1,
                temperature=self.temperature,
                messages=messages,
                extra_body=self.extra_body,
                timeout=600,
            )

            try:
                if self.verbose:
                    self.logger.info('Start calling OpenAI API')
                responses = await self.openai_client.chat.completions.create(**query_data)

                if self.verbose:
                    self.logger.info(
                        'Successfully get response from OpenAI API')
                    try:
                        self.logger.info(responses)
                    except Exception as e:  # noqa F841
                        pass
                if not responses.choices:
                    self.logger.error(
                        'Response is empty, it is an internal server error \
                            from the API provider.')
                return index, responses.choices[0].message.content

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
                    self.logger.info(f'Status Code: {status_code},\n'
                                     f'Original Error Message: {e},\n'
                                     f'Return Message: {error_message} ')
                    return index, error_message
                else:
                    self.logger.warning(f"Failed to get response for {e}, retry {num_retries}/{self.retry}")
            except Exception as e:
                self.logger.warning(f"Failed to get response for {e}, retry {num_retries}/{self.retry}")
            num_retries += 1
        raise RuntimeError('Calling OpenAI API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')

    def _get_state_key(self, api_base: str, model_name: str):
        return api_base + model_name

    def bin_trim(self, prompt: str, num_token: int, mode: Literal['front', 'mid', 'rear']) -> str:
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
            # mode: Literal['front', 'mid', 'rear'] = self.mode
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

        if self.mode == 'front':
            prompt = sep.join(words[-l:])
        elif self.mode == 'mid':
            prompt = sep.join(words[:l]) + sep.join(words[-l:])
        elif self.mode == 'rear':
            prompt = sep.join(words[:l])
        return prompt


