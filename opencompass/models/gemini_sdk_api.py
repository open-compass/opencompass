import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from opencompass.registry import MODELS
from opencompass.utils import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, List[Dict], str]

ROLE_MAPPING = {
    'HUMAN': 'user',
    'BOT': 'model',
    'SYSTEM': 'system',
    'human': 'user',
    'bot': 'model',
    'system': 'system',
    'user': 'user',
    'assistant': 'model',
    'model': 'model',
}


@MODELS.register_module()
class GeminiSDK(BaseAPIModel):
    """Model wrapper around Google Gen AI SDK.

    Args:
        key (str): Authorization key. If set to "ENV", use GOOGLE_API_KEY or
            GEMINI_API_KEY from environment. GOOGLE_API_KEY has higher
            priority.
        path (str): The Gemini model to be used.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 2.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt template.
        temperature (float, optional): Sampling temperature. Defaults to 0.0.
        top_p (float, optional): Nucleus sampling parameter.
        top_k (float, optional): Top-k sampling parameter.
        thinking (Dict, optional): Gemini thinking config passed to
            google.genai.types.ThinkingConfig. For Gemini 2.5, use fields
            such as include_thoughts and thinking_budget.
        base_url (str, optional): Gemini SDK base URL. If omitted, use
            GEMINI_BASE_URL from environment.
        stream (bool): Whether to use streaming response. Defaults to False.
        retry (int): Number of retries if the API call fails. Defaults to 2.
        think_tag (str): Separator used between thought summary content and
            final text. Defaults to '</think>'.
        gemini_extra_kwargs (Dict, optional): Extra keyword arguments passed
            to google.genai.types.GenerateContentConfig.
        client_extra_kwargs (Dict, optional): Extra keyword arguments passed
            to google.genai.Client.
        max_workers (int, optional): Maximum number of worker threads for
            concurrent API requests.
    """

    def __init__(
        self,
        key: Optional[str] = 'ENV',
        path: str = 'gemini-2.5-flash',
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        thinking: Optional[Dict] = None,
        base_url: Optional[str] = None,
        stream: bool = False,
        retry: int = 2,
        think_tag: str = '</think>',
        gemini_extra_kwargs: Optional[Dict] = None,
        client_extra_kwargs: Optional[Dict] = None,
        max_workers: Optional[int] = None,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
        )
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError('Import google.genai failed. Please install it '
                              'with "pip install google-genai" and try again.')

        key = self._resolve_key(key)
        base_url = self._resolve_base_url(base_url)

        client_kwargs = {'api_key': key}
        if base_url:
            client_kwargs['http_options'] = types.HttpOptions(
                base_url=base_url)
        if client_extra_kwargs:
            client_kwargs.update(client_extra_kwargs)

        self.genai = genai.Client(**client_kwargs)
        self.genai_types = types
        self.model = path
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.thinking = thinking
        self.base_url = base_url
        self.stream = stream
        self.think_tag = think_tag
        self.gemini_extra_kwargs = gemini_extra_kwargs
        self.max_workers = max_workers

    @staticmethod
    def _resolve_key(key: Optional[str] = 'ENV') -> str:
        """Resolve Gemini SDK API key from explicit args or env vars."""
        if key not in (None, 'ENV'):
            return key

        resolved_key = os.getenv('GOOGLE_API_KEY') or os.getenv(
            'GEMINI_API_KEY')
        if not resolved_key:
            raise ValueError('GOOGLE_API_KEY or GEMINI_API_KEY must be set '
                             'when key is "ENV".')
        return resolved_key

    @staticmethod
    def _resolve_base_url(base_url: Optional[str] = None) -> Optional[str]:
        """Resolve Gemini SDK base URL from explicit args or env vars."""
        return base_url or os.getenv('GEMINI_BASE_URL')

    @staticmethod
    def _convert_role(role: str) -> str:
        """Convert OpenCompass or raw prompt roles to Gemini roles."""
        if role not in ROLE_MAPPING:
            raise ValueError(f'Unsupported role: {role}')
        return ROLE_MAPPING[role]

    @staticmethod
    def _message_content(item: Dict) -> str:
        """Read content from OpenCompass or raw prompt message dicts."""
        if 'prompt' in item:
            return item['prompt']
        if 'content' in item:
            return item['content']
        raise KeyError(f'Message item missing prompt/content: {item}')

    @classmethod
    def _convert_prompt_list(
        cls,
        input: Union[PromptList, List[Dict]],
    ) -> Tuple[List[Dict], List[str]]:
        """Convert PromptList/raw messages to Gemini SDK contents."""
        contents = []
        system_prompts = []
        for item in input:
            if not isinstance(item, dict):
                raise TypeError(
                    f'Prompt item must be a dict, got {type(item)}')
            role = cls._convert_role(item['role'])
            content = cls._message_content(item)
            if role == 'system':
                system_prompts.append(content)
                continue
            contents.append({
                'role': role,
                'parts': [{
                    'text': content,
                }],
            })
        return contents, system_prompts

    @staticmethod
    def _block_attr(block, name: str, default=None):
        """Read a value from a Google Gen AI SDK object or dict block."""
        if isinstance(block, dict):
            return block.get(name, default)
        return getattr(block, name, default)

    @classmethod
    def _parts_thinking_and_text(cls, parts) -> Tuple[str, str]:
        """Extract thought summaries and final text from response parts."""
        thinking_chunks = []
        text_chunks = []
        for part in parts or []:
            text = cls._block_attr(part, 'text')
            if text:
                if cls._block_attr(part, 'thought', False):
                    thinking_chunks.append(text)
                else:
                    text_chunks.append(text)
        return ''.join(thinking_chunks), ''.join(text_chunks)

    @staticmethod
    def _merge_thinking_and_text(thinking: str, text: str,
                                 think_tag: str) -> str:
        """Join Gemini thought summary content with final text."""
        if thinking:
            if text:
                return thinking + think_tag + text
            return thinking
        return text

    @classmethod
    def _extract_response_texts(cls, response) -> Tuple[str, str]:
        """Extract thought summary/final text from a response or chunk."""
        candidates = cls._block_attr(response, 'candidates') or []
        if candidates:
            content = cls._block_attr(candidates[0], 'content')
            parts = cls._block_attr(content, 'parts') if content else []
            if parts:
                return cls._parts_thinking_and_text(parts)

        parts = cls._block_attr(response, 'parts')
        if parts:
            return cls._parts_thinking_and_text(parts)

        try:
            text = getattr(response, 'text')
        except Exception:
            text = None
        if text:
            return '', text
        if text == '':
            return '', ''
        return '', ''

    @classmethod
    def _parse_response(cls,
                        response,
                        think_tag: str = '</think>',
                        strip: bool = True) -> str:
        """Extract text and optional thought summaries from an SDK response."""
        thinking, text = cls._extract_response_texts(response)
        parsed = cls._merge_thinking_and_text(thinking, text, think_tag)
        if parsed:
            return parsed.strip() if strip else parsed
        return ''

    @classmethod
    def _parse_stream_response(cls,
                               responses,
                               think_tag: str = '</think>') -> str:
        """Accumulate thought summary/text deltas from a stream response."""
        thinking_chunks = []
        text_chunks = []
        for response in responses:
            thinking, text = cls._extract_response_texts(response)
            if thinking:
                thinking_chunks.append(thinking)
            if text:
                text_chunks.append(text)
        return cls._merge_thinking_and_text(''.join(thinking_chunks),
                                            ''.join(text_chunks),
                                            think_tag).strip()

    def _build_thinking_config(self):
        """Build google.genai ThinkingConfig from configured kwargs."""
        if self.thinking is None:
            return None
        if isinstance(self.thinking, dict):
            return self.genai_types.ThinkingConfig(**self.thinking)
        return self.thinking

    def _build_config(self, max_out_len: int, system_prompts: List[str]):
        """Build google.genai GenerateContentConfig for one request."""
        config_kwargs = {'candidate_count': 1}
        if max_out_len is not None:
            config_kwargs['max_output_tokens'] = max_out_len
        if self.temperature is not None:
            config_kwargs['temperature'] = self.temperature
        if self.top_p is not None:
            config_kwargs['top_p'] = self.top_p
        if self.top_k is not None:
            config_kwargs['top_k'] = self.top_k
        if system_prompts:
            config_kwargs['system_instruction'] = '\n'.join(system_prompts)
        thinking_config = self._build_thinking_config()
        if thinking_config is not None:
            config_kwargs['thinking_config'] = thinking_config
        if self.gemini_extra_kwargs:
            config_kwargs.update(self.gemini_extra_kwargs)
        return self.genai_types.GenerateContentConfig(**config_kwargs)

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
        if len(inputs) == 1:
            # Forget multi-thread for single inference.
            return [self._generate(inputs[0], max_out_len)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(self._generate, inputs,
                                 [max_out_len] * len(inputs)),
                    total=len(inputs),
                    desc='Inferencing',
                ))
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        """Generate results given an input."""
        assert isinstance(input, (str, list, PromptList))

        if isinstance(input, str):
            contents = [{
                'role': 'user',
                'parts': [{
                    'text': input,
                }],
            }]
            system_prompts = []
        else:
            contents, system_prompts = self._convert_prompt_list(input)

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                api_params = {
                    'model': self.model,
                    'contents': contents,
                    'config': self._build_config(max_out_len, system_prompts),
                }
                if self.stream:
                    responses = self.genai.models.generate_content_stream(
                        **api_params)
                    return self._parse_stream_response(responses,
                                                       self.think_tag)

                response = self.genai.models.generate_content(**api_params)
                return self._parse_response(response, self.think_tag)
            except Exception as e:
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError('Calling Gemini SDK API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')
