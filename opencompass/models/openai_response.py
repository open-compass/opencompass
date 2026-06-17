import random
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .openai_api import OPENAISDK_API_BASE, OpenAI, set_proxy_cfg

PromptType = Union[PromptList, str, List[Dict[str, Any]]]
RESPONSE_ROLE = {'system', 'user', 'assistant', 'developer'}
LOGPROBS_INCLUDE = 'message.output_text.logprobs'


@MODELS.register_module()
class OpenAIResponse(OpenAI):
    """OpenAI Responses API model wrapper.

    This class keeps the OpenCompass generation interface while calling
    ``client.responses.create`` underneath.

    Args:
        path (str): OpenAI model name.
        max_seq_len (int): Maximum input plus output token budget.
        query_per_second (int): Request rate limit.
        retry (int): Number of retries after API failures.
        key (str or List[str]): OpenAI API key(s), or ``'ENV'`` for
            ``OPENAI_API_KEY``.
        org (str or List[str], optional): OpenAI organization(s).
        meta_template (Dict, optional): OpenCompass API model meta template.
        openai_api_base (str or List[str]): SDK base URL, usually
            ``https://api.openai.com/v1/``.
        openai_proxy_url (str, optional): Optional HTTP proxy URL.
        mode (str): Input truncation mode, one of ``none``, ``front``,
            ``mid`` or ``rear``.
        logprobs (bool, optional): Whether to request output token logprobs.
        top_logprobs (int, optional): Number of top token logprobs to return.
        temperature (float, optional): Temperature override.
        tokenizer_path (str, optional): Tokenizer model or path.
        extra_body (Dict, optional): Extra body passed through the OpenAI SDK.
        http_client_cfg (Dict, optional): Extra ``httpx.Client`` kwargs.
        status_code_mappings (Dict, optional): Status-code fallback outputs.
        openai_extra_kwargs (Dict, optional): Direct ``responses.create``
            keyword arguments.
        response_kwargs (Dict, optional): Direct ``responses.create`` keyword
            arguments. Kept as an alias for ``openai_extra_kwargs``.
        include_reasoning_content (bool): If true, attempt to include exposed
            reasoning text before the final answer.
        timeout (int): Request timeout in seconds.
    """

    is_api: bool = True

    def __init__(
        self,
        path: str = 'gpt-4.1',
        max_seq_len: int = 16384,
        query_per_second: int = 1,
        rpm_verbose: bool = False,
        retry: int = 2,
        key: Union[str, List[str]] = 'ENV',
        org: Optional[Union[str, List[str]]] = None,
        meta_template: Optional[Dict] = None,
        openai_api_base: Union[str, List[str]] = OPENAISDK_API_BASE,
        openai_proxy_url: Optional[str] = None,
        mode: str = 'none',
        logprobs: Optional[bool] = False,
        top_logprobs: Optional[int] = None,
        temperature: Optional[float] = None,
        tokenizer_path: Optional[str] = None,
        extra_body: Optional[Dict] = None,
        verbose: bool = False,
        http_client_cfg: Optional[Dict] = None,
        status_code_mappings: Optional[Dict[int, str]] = None,
        think_tag: str = '</think>',
        max_workers: Optional[int] = None,
        openai_extra_kwargs: Optional[Dict] = None,
        response_kwargs: Optional[Dict] = None,
        include_reasoning_content: bool = False,
        timeout: int = 3600,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            rpm_verbose=rpm_verbose,
            retry=retry,
            key=key,
            org=org,
            meta_template=meta_template,
            openai_api_base=openai_api_base,
            openai_proxy_url=openai_proxy_url,
            mode=mode,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            temperature=temperature,
            tokenizer_path=tokenizer_path,
            extra_body=extra_body,
            verbose=verbose,
            think_tag=think_tag,
            max_workers=max_workers,
        )
        if isinstance(openai_api_base, list):
            self.openai_api_base = random.choice(openai_api_base)
        else:
            self.openai_api_base = openai_api_base

        self.http_client_cfg = dict(http_client_cfg or {})
        self.status_code_mappings = dict(status_code_mappings or {})
        self.openai_extra_kwargs = dict(response_kwargs or {})
        if openai_extra_kwargs:
            self.openai_extra_kwargs.update(openai_extra_kwargs)
        # Backward-compatible alias for configs using the first implementation.
        self.response_kwargs = self.openai_extra_kwargs
        self.include_reasoning_content = include_reasoning_content
        self.timeout = timeout
        self.openai_client = self._create_fresh_client()
        if self.verbose:
            self.logger.info(f'Used openai_client: {self.openai_client}')

    def _create_fresh_client(self):
        """Create an OpenAI SDK client for the Responses API."""
        from openai import OpenAI as OpenAIClient

        current_key = self._next_valid_key()
        http_client_cfg = self.http_client_cfg.copy()
        set_proxy_cfg(http_client_cfg, self.proxy_url)
        limits = httpx.Limits(max_keepalive_connections=2048,
                              max_connections=4096)
        http_client = httpx.Client(
            **http_client_cfg,
            timeout=httpx.Timeout(self.timeout),
            limits=limits,
        )
        return OpenAIClient(base_url=self.openai_api_base,
                            api_key=current_key,
                            http_client=http_client)

    def _generate(
        self,
        input: PromptType,
        max_out_len: int,
        temperature: float,
    ) -> str:
        """Generate one output with the OpenAI Responses API."""
        from openai import APIStatusError, BadRequestError

        assert isinstance(input, (str, list, PromptList))

        messages, max_out_len = self._preprocess_messages(
            input, max_out_len, self.max_seq_len, self.mode,
            self.get_token_len)

        query_data = self._build_query_data(messages, max_out_len, temperature)

        num_retries = 0
        while num_retries < self.retry:
            self.acquire()
            try:
                if self.verbose:
                    self.logger.info('Start calling OpenAI Responses API')
                response = self.openai_client.responses.create(
                    **query_data, timeout=self.timeout)
                if self.verbose:
                    self.logger.info(
                        'Successfully get response from OpenAI Responses API '
                        'with query: %s', query_data)
                    try:
                        self.logger.info(response)
                    except Exception:
                        pass
                text = self._extract_response_text(response)

                if text:
                    return text

                incomplete_reason = self._get_nested_value(
                    response, 'incomplete_details', 'reason')
                if incomplete_reason == 'content_filter':
                    self.logger.info('Response is filtered by content_filter.')
                    return ''

                error = self._get_value(response, 'error')
                if error:
                    self.logger.error('Responses API returned error: %s',
                                      error)
                    num_retries += 1
                    continue

                status = self._get_value(response, 'status')
                if status == 'completed':
                    self.logger.info(
                        'Server does not return any content and response '
                        'status is <completed>, the input query is: %s',
                        query_data)
                    return ''

                self.logger.error(
                    'Failed to extract content from the response. Please '
                    'check the API response for detail information. API '
                    'response: %s', response)
                num_retries += 1
                continue
            except (BadRequestError, APIStatusError) as e:
                status_code = getattr(e, 'status_code', None)
                if (status_code is not None
                        and status_code in self.status_code_mappings):
                    error_message = self.status_code_mappings[status_code]
                    self.logger.error(
                        f'error occurs at {self.openai_api_base}')
                    self.logger.info(f'Status Code: {status_code}, \n'
                                     f'Original Error Message: {e}, \n'
                                     f'Return Message: {error_message} ')
                    return error_message
                self.logger.error(f'error occurs at {self.openai_api_base}')
                self.logger.error(e)
            except Exception as e:
                self.logger.error(f'error occurs at {self.openai_api_base}')
                self.logger.error(e)
            finally:
                self.release()
            num_retries += 1

        raise RuntimeError('Calling OpenAI Responses API failed after '
                           f'retrying for {self.retry} times. Check the logs '
                           'for details.')

    def _preprocess_messages(
        self,
        input: PromptType,
        max_out_len: int,
        max_seq_len: int,
        mode: str,
        get_token_len_func,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """Preprocess input and additionally support Responses-only roles."""
        if self._is_response_message_list(input):
            return self._preprocess_response_messages(input, max_out_len,
                                                      max_seq_len, mode,
                                                      get_token_len_func)
        return super()._preprocess_messages(input, max_out_len, max_seq_len,
                                            mode, get_token_len_func)

    def _preprocess_response_messages(
        self,
        input: List[Dict[str, Any]],
        max_out_len: int,
        max_seq_len: int,
        mode: str,
        get_token_len_func,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        messages = [message.copy() for message in input]
        input_len = sum(
            self._content_token_len(message.get(
                'content', ''), get_token_len_func) for message in messages)

        if mode == 'none' and input_len > max_seq_len:
            raise ValueError(
                f'Input length ({input_len}) exceeds max_seq_len '
                f'({max_seq_len}) and mode is set to "none". Please either '
                f'change the mode or increase the max_seq_len.')

        if mode != 'none':
            trim_length = max_seq_len - 100
            if max_out_len is not None:
                trim_length -= max_out_len
            for message in messages:
                message['content'] = self._trim_response_content(
                    message.get('content', ''), trim_length, mode)
            input_len = sum(
                self._content_token_len(message.get('content', ''),
                                        get_token_len_func)
                for message in messages)

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

    def _build_query_data(
        self,
        messages: List[Dict[str, Any]],
        max_out_len: Optional[int],
        temperature: Optional[float],
    ) -> Dict[str, Any]:
        query_data = dict(
            model=self.path,
            input=self._messages_to_response_input(messages),
        )
        if max_out_len is not None:
            query_data['max_output_tokens'] = max_out_len
        if temperature is not None:
            query_data['temperature'] = temperature
        if self.extra_body:
            query_data['extra_body'] = self.extra_body
        if self.openai_extra_kwargs:
            query_data.update(self.openai_extra_kwargs)
        if self.logprobs:
            self._append_include(query_data, LOGPROBS_INCLUDE)
        if self.top_logprobs is not None and 'top_logprobs' not in query_data:
            query_data['top_logprobs'] = self.top_logprobs
        return query_data

    def _messages_to_response_input(self, messages: List[Dict[str,
                                                              Any]]) -> List:
        """Convert OpenCompass chat messages into Responses API input."""
        response_input = []
        for message in messages:
            response_input.append(message.copy())
        return response_input

    def _is_response_message_list(self, input: Any) -> bool:
        return isinstance(input, list) and len(input) > 0 and all(
            isinstance(item, dict) and item.get('role') in RESPONSE_ROLE
            and 'content' in item for item in input)

    def _content_token_len(self, content: Any, get_token_len_func) -> int:
        if isinstance(content, str):
            return get_token_len_func(content)
        if isinstance(content, list):
            total = 0
            for part in content:
                if isinstance(part, str):
                    total += get_token_len_func(part)
                elif isinstance(part, dict):
                    text = part.get('text')
                    if isinstance(text, str):
                        total += get_token_len_func(text)
            return total
        return 0

    def _trim_response_content(self, content: Any, num_token: int,
                               mode: str) -> Any:
        if isinstance(content, str):
            return self._bin_trim(content, num_token, mode)
        if isinstance(content, list):
            trimmed_content = []
            for part in content:
                if isinstance(part, dict):
                    trimmed_part = part.copy()
                    text = trimmed_part.get('text')
                    if isinstance(text, str):
                        trimmed_part['text'] = self._bin_trim(
                            text, num_token, mode)
                    trimmed_content.append(trimmed_part)
                else:
                    trimmed_content.append(part)
            return trimmed_content
        return content

    @staticmethod
    def _append_include(query_data: Dict[str, Any], include: str) -> None:
        existing_include = query_data.get('include')
        if existing_include is None:
            query_data['include'] = [include]
        elif isinstance(existing_include, str):
            if existing_include != include:
                query_data['include'] = [existing_include, include]
        else:
            include_values = list(existing_include)
            if include not in include_values:
                include_values.append(include)
            query_data['include'] = include_values

    def _extract_response_text(self, response: Any) -> str:
        """Extract final answer text from a Responses API response object."""
        output_text = self._get_value(response, 'output_text')
        if isinstance(output_text, str) and not self.include_reasoning_content:
            return output_text

        reasoning_text, message_text = self._extract_output_items(response)
        if not message_text and isinstance(output_text, str):
            message_text = output_text

        if self.include_reasoning_content and reasoning_text:
            if message_text:
                return reasoning_text + self.think_tag + message_text
            return reasoning_text
        return message_text

    def _extract_output_items(self, response: Any) -> Tuple[str, str]:
        output = self._get_value(response, 'output', []) or []
        reasoning_chunks = []
        message_chunks = []

        for item in output:
            item_type = self._get_value(item, 'type')
            if item_type == 'message':
                message_chunks.extend(self._extract_content_text(item))
            elif item_type == 'reasoning':
                reasoning_chunks.extend(self._extract_reasoning_text(item))

        return ''.join(reasoning_chunks), ''.join(message_chunks)

    def _extract_content_text(self, item: Any) -> List[str]:
        content = self._get_value(item, 'content', []) or []
        if isinstance(content, str):
            return [content]

        chunks = []
        for part in content:
            part_type = self._get_value(part, 'type')
            text = self._get_value(part, 'text')
            if isinstance(text, str) and part_type in {
                    'output_text',
                    'text',
                    None,
            }:
                chunks.append(text)
        return chunks

    def _extract_reasoning_text(self, item: Any) -> List[str]:
        chunks = []
        for field in ('summary', 'content'):
            value = self._get_value(item, field, []) or []
            if isinstance(value, str):
                chunks.append(value)
                continue
            for part in value:
                text = self._get_value(part, 'text')
                if isinstance(text, str):
                    chunks.append(text)
        return chunks

    def _get_nested_value(self, obj: Any, *keys: str) -> Any:
        value = obj
        for key in keys:
            value = self._get_value(value, key)
            if value is None:
                return None
        return value

    @staticmethod
    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
