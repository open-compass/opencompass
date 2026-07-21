"""LiteLLM AI gateway backend for OpenCompass.

Routes ``generate`` through ``litellm.completion()`` to 100+ providers (OpenAI,
Anthropic, Bedrock, Azure, Vertex, Gemini, Ollama, OpenRouter, Groq, DeepSeek,
etc.) using provider-native API keys. The target model is selected by its
LiteLLM-style prefix, e.g. ``anthropic/claude-3-5-sonnet-20241022``,
``azure/gpt-4o``, ``bedrock/anthropic.claude-3-sonnet-20240229-v1:0``,
``ollama/llama3``.

See https://docs.litellm.ai/docs/providers for the full list of supported
providers and their model-name prefix convention.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, List, str]

CHATML_ROLE = ['system', 'user', 'assistant']
CORE_CALL_KWARGS = {'model', 'messages', 'max_tokens', 'drop_params', 'stream'}
RESERVED_OUTPUT_TOKENS = 100


@MODELS.register_module()
class LiteLLMAPI(BaseAPIModel):
    """Model wrapper that dispatches to ``litellm.completion``.

    Args:
        path (str): LiteLLM-style fully-qualified model name, e.g.
            ``anthropic/claude-3-5-sonnet-20241022`` or ``azure/gpt-4o``.
            Forwarded directly as ``litellm.completion(model=...)``.
        key (str, optional): Provider API key. When ``None`` (default), LiteLLM
            resolves credentials from provider-specific env vars
            (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``,
            ``AWS_*``, ``GROQ_API_KEY``, ...) based on the model prefix.
        api_base (str, optional): Custom base URL. Forwarded as LiteLLM's
            ``api_base`` kwarg. Useful for self-hosted endpoints, proxies,
            and Azure deployments.
        api_version (str, optional): For Azure OpenAI deployments (e.g.
            ``2025-01-01-preview``). Forwarded as LiteLLM's ``api_version``.
        query_per_second (int): Max queries per second. Defaults to 2.
        rpm_verbose (bool): Whether to log throttling. Defaults to False.
        max_seq_len (int): Max sequence length (used by the BaseAPIModel
            plumbing for length bookkeeping). Defaults to 16384.
        meta_template (Dict, optional): Standard OpenCompass meta template.
        retry (int): Retries on transient failure. Defaults to 2.
        system_prompt (str): Optional system message prepended when no
            system message is already present. Defaults to ``''``.
        temperature (float, optional): Sampling temperature. If unset, LiteLLM
            uses the per-provider default.
        stream (bool): Whether to request streaming responses from LiteLLM.
            Defaults to False.
        think_tag (str): Tag inserted between reasoning content and final
            answer when LiteLLM returns ``reasoning_content``. Defaults to
            ``'</think>'``.
        max_workers (int, optional): Size of the ThreadPoolExecutor for batch
            dispatch. Defaults to ``None`` (Python default).
        extra_body (Dict, optional): Arbitrary extra kwargs forwarded to
            ``litellm.completion``, e.g. ``{"reasoning_effort": "high"}``.
    """

    is_api: bool = True

    def __init__(
        self,
        path: str,
        key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        query_per_second: int = 2,
        rpm_verbose: bool = False,
        max_seq_len: int = 16384,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        system_prompt: str = '',
        temperature: Optional[float] = None,
        stream: bool = False,
        think_tag: str = '</think>',
        max_workers: Optional[int] = None,
        extra_body: Optional[Dict] = None,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            rpm_verbose=rpm_verbose,
            meta_template=meta_template,
            retry=retry,
        )
        self.key = key
        self.api_base = api_base
        self.api_version = api_version
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.stream = stream
        self.think_tag = think_tag
        self.max_workers = max_workers
        self.extra_body = extra_body or {}

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
        **gen_kwargs,
    ) -> List[str]:
        """Generate responses for a batch of inputs.

        Args:
            inputs: list of strings or ``PromptList`` messages.
            max_out_len: max output tokens per response. Defaults to 512.
            **gen_kwargs: extra per-call generation kwargs forwarded to
                LiteLLM, except core request fields managed by this wrapper.

        Returns:
            list of generated strings, one per input.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [gen_kwargs] * len(inputs)))
        self.flush()
        return results

    def _build_messages(self, input: PromptType) -> List[Dict[str, str]]:
        """Convert an OpenCompass input into OpenAI-shaped messages.

        Handles three cases:
            1. plain ``str`` -> wrap as a single ``user`` message.
            2. ``PromptList`` of CHATML-shaped dicts (role in
               CHATML_ROLE, key ``'content'``) -> pass through.
            3. ``PromptList`` of OpenCompass-native dicts (role in
               {'SYSTEM','HUMAN','BOT'}, key ``'prompt'``) -> translate roles
               to OpenAI convention and rename the content key.

        Prepends ``self.system_prompt`` as a system message iff no system
        message is already present in the result.
        """
        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        elif (isinstance(input, list) and len(input) > 0 and all(
                isinstance(item, dict) and 'role' in item
                and item['role'] in CHATML_ROLE for item in input)):
            # Already CHATML-shaped, assume ``content`` key is populated.
            messages = [{
                'role': item['role'],
                'content': item.get('content', item.get('prompt', '')),
            } for item in input]
        else:
            messages = []
            assert isinstance(input,
                              list), ('Expected str or list-like prompt, got '
                                      f'{type(input).__name__}: {input!r}')
            for item in input:
                if isinstance(item, str):
                    messages.append({'role': 'user', 'content': item})
                    continue
                role = item.get('role', '')
                content = item.get('prompt', item.get('content', ''))
                if role == 'HUMAN':
                    messages.append({'role': 'user', 'content': content})
                elif role == 'BOT':
                    messages.append({'role': 'assistant', 'content': content})
                elif role == 'SYSTEM':
                    messages.append({'role': 'system', 'content': content})
                else:
                    # Unknown role — default to user to avoid losing content.
                    messages.append({'role': 'user', 'content': content})

        if self.system_prompt and not any(m['role'] == 'system'
                                          for m in messages):
            messages.insert(0, {
                'role': 'system',
                'content': self.system_prompt,
            })

        return messages

    def _get_messages_token_len(self, messages: List[Dict[str, str]]) -> int:
        return sum(
            self.get_token_len(str(message.get('content', '')))
            for message in messages)

    def _adjust_max_out_len(self, messages: List[Dict[str, str]],
                            max_out_len: int) -> int:
        input_len = self._get_messages_token_len(messages)
        if input_len > self.max_seq_len:
            raise ValueError(
                f'Input length ({input_len}) exceeds max_seq_len '
                f'({self.max_seq_len}). Please increase max_seq_len or '
                'shorten the prompt.')

        available_out_len = (self.max_seq_len - input_len -
                             RESERVED_OUTPUT_TOKENS)
        if available_out_len <= 0:
            raise ValueError(
                f'max_out_len ({available_out_len}) is less than or equal '
                f'to 0. This may be due to input length ({input_len}) being '
                f'too close to max_seq_len ({self.max_seq_len}). Please '
                'increase max_seq_len or shorten the prompt.')

        adjusted_max_out_len = min(max_out_len, available_out_len)
        if adjusted_max_out_len < max_out_len:
            self.logger.warning(
                f'max_out_len was truncated from {max_out_len} to '
                f'{adjusted_max_out_len} due to input length')
        return adjusted_max_out_len

    def _build_call_kwargs(self,
                           messages: List[Dict[str, str]],
                           max_out_len: int,
                           gen_kwargs: Optional[Dict] = None) -> Dict:
        gen_kwargs = gen_kwargs or {}
        blocked_kwargs = sorted(set(gen_kwargs) & CORE_CALL_KWARGS)
        if blocked_kwargs:
            self.logger.warning(
                'Ignoring LiteLLM generation kwargs that would override '
                f'core request fields: {blocked_kwargs}')
        safe_gen_kwargs = {
            key: value
            for key, value in gen_kwargs.items() if key not in CORE_CALL_KWARGS
        }
        safe_extra_body = {
            key: value
            for key, value in self.extra_body.items()
            if key not in CORE_CALL_KWARGS
        }
        call_kwargs: Dict = {
            **safe_extra_body,
            **safe_gen_kwargs,
            'model': self.path,
            'messages': messages,
            'max_tokens': max_out_len,
            'drop_params': True,
        }
        if self.temperature is not None:
            call_kwargs['temperature'] = self.temperature
        if self.key is not None:
            call_kwargs['api_key'] = self.key
        if self.api_base is not None:
            call_kwargs['api_base'] = self.api_base
        if self.api_version is not None:
            call_kwargs['api_version'] = self.api_version
        if self.stream:
            call_kwargs['stream'] = True
        return call_kwargs

    @staticmethod
    def _get_field(obj: Any, field: str):
        if isinstance(obj, dict):
            return obj.get(field)
        return getattr(obj, field, None)

    def _get_reasoning_content(self, message: Any) -> str:
        reasoning_content = self._get_field(message, 'reasoning_content')
        if reasoning_content:
            return reasoning_content

        thinking_blocks = self._get_field(message, 'thinking_blocks') or []
        chunks: List[str] = []
        for block in thinking_blocks:
            block_type = self._get_field(block, 'type')
            if block_type and block_type != 'thinking':
                continue
            thinking = self._get_field(block, 'thinking')
            if thinking:
                chunks.append(thinking)
        return ''.join(chunks)

    def _merge_reasoning_content(self, content: Optional[str],
                                 reasoning_content: Optional[str]) -> str:
        content = content or ''
        reasoning_content = reasoning_content or ''
        if not reasoning_content:
            return content
        if content:
            return reasoning_content + self.think_tag + content
        return reasoning_content

    def _parse_stream_response(self, response) -> str:
        content_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        for chunk in response:
            choices = self._get_field(chunk, 'choices') or []
            if not choices:
                continue
            delta = self._get_field(choices[0], 'delta')
            reasoning_content = self._get_field(delta, 'reasoning_content')
            if reasoning_content:
                reasoning_chunks.append(reasoning_content)
            content = self._get_field(delta, 'content')
            if content:
                content_chunks.append(content)
        return self._merge_reasoning_content(''.join(content_chunks),
                                             ''.join(reasoning_chunks))

    def _generate(self,
                  input: PromptType,
                  max_out_len: int,
                  gen_kwargs: Optional[Dict] = None) -> str:
        try:
            import litellm
        except ImportError as err:
            raise ImportError('litellm is not installed. Install with: '
                              'pip install litellm') from err

        assert isinstance(input, (str, list))

        messages = self._build_messages(input)
        max_out_len = self._adjust_max_out_len(messages, max_out_len)
        call_kwargs = self._build_call_kwargs(messages, max_out_len,
                                              gen_kwargs)

        num_retries = 0
        last_err: Optional[BaseException] = None
        while num_retries < self.retry:
            self.acquire()
            try:
                response = litellm.completion(**call_kwargs)
                if self.stream:
                    return self._parse_stream_response(response)
            except Exception as err:  # noqa: BLE001
                last_err = err
                num_retries += 1
                self.logger.warning(
                    f'LiteLLM request failed '
                    f'(attempt {num_retries}/{self.retry}): {err}')
                time.sleep(1)
                continue
            finally:
                self.release()

            try:
                message = response.choices[0].message
                content = self._get_field(message, 'content')
                reasoning_content = self._get_reasoning_content(message)
                # Some providers return ``None`` on filtered / empty output.
                return self._merge_reasoning_content(content,
                                                     reasoning_content)
            except (AttributeError, IndexError, TypeError) as err:
                last_err = err
                num_retries += 1
                self.logger.warning(
                    f'LiteLLM response parse failed '
                    f'(attempt {num_retries}/{self.retry}): {err}')
                time.sleep(1)

        raise RuntimeError(
            f'LiteLLM request failed after {self.retry} retries: {last_err}')
