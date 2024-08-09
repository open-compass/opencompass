from typing import Dict, List, Union

from opencompass.models import OpenAI
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]
OPENAI_API_BASE = 'https://api.openai.com/v1/chat/completions'


class OpenAISDK(OpenAI):

    def __init__(
        self,
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
        extra_body: Dict | None = None,
    ):
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
            mode,
            logprobs,
            top_logprobs,
            temperature,
            tokenizer_path,
            extra_body,
        )
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
        try:
            max_out_len = min(
                max_out_len,
                context_window - self.get_token_len(str(input)) - 100)
        except KeyError:
            max_out_len = max_out_len
        if max_out_len <= 0:
            return ''
        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                responses = self.opeanai_cleint.chat.completions.create(
                    model=self.path,
                    max_tokens=max_out_len,
                    n=1,
                    temperature=self.temperature,
                    messages=messages,
                )
                return responses.choices[0].message.content
            except Exception as e:
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError('Calling OpenAI API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')
