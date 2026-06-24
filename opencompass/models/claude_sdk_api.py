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
    'BOT': 'assistant',
    'SYSTEM': 'system',
    'human': 'user',
    'bot': 'assistant',
    'system': 'system',
    'user': 'user',
    'assistant': 'assistant',
}


@MODELS.register_module()
class ClaudeSDK(BaseAPIModel):
    """Model wrapper around Claude SDK API.

    Args:
        key (str): Authorization key. If set to "ENV", use ANTHROPIC_API_KEY
            from environment.
        path (str): The model to be used. Defaults to claude-2.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        base_url (str, optional): Anthropic-compatible base URL. If omitted,
            use ANTHROPIC_BASE_URL from environment.
        stream (bool): Whether to use streaming response. Defaults to False.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        claude_extra_kwargs (Dict, optional): Extra keyword arguments passed
            to anthropic.messages.create.
    """

    def __init__(
        self,
        key: str,
        path: str = 'claude-2',
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        temperature: Optional[float] = 0.0,
        thinking: Optional[Dict] = None,
        base_url: Optional[str] = None,
        stream: bool = False,
        retry: int = 2,
        claude_extra_kwargs: Optional[Dict] = None,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError('Import anthropic failed. Please install it '
                              'with "pip install anthropic" and try again.')

        if key == 'ENV':
            key = os.getenv('ANTHROPIC_API_KEY')
            if not key:
                raise ValueError('ANTHROPIC_API_KEY must be set when key is '
                                 '"ENV".')

        base_url = self._resolve_base_url(base_url)
        client_kwargs = {'api_key': key}
        if base_url:
            client_kwargs['base_url'] = base_url

        self.anthropic = Anthropic(**client_kwargs)
        self.model = path
        self.temperature = temperature
        self.thinking = thinking
        self.base_url = base_url
        self.stream = stream
        self.claude_extra_kwargs = claude_extra_kwargs

    @staticmethod
    def _resolve_base_url(base_url: Optional[str] = None) -> Optional[str]:
        """Resolve Anthropic SDK base URL from explicit args or env vars."""
        return base_url or os.getenv('ANTHROPIC_BASE_URL')

    @staticmethod
    def _convert_role(role: str) -> str:
        """Convert OpenCompass or raw prompt roles to Anthropic roles."""
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
        """Convert PromptList/raw messages to Anthropic message params."""
        messages = []
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
            messages.append({'role': role, 'content': content})
        return messages, system_prompts

    @staticmethod
    def _parse_stream_response(responses) -> str:
        """Accumulate text deltas from an Anthropic streaming response."""
        text_chunks = []
        for event in responses:
            if isinstance(event, dict):
                delta = event.get('delta') or {}
                text = delta.get('text') if isinstance(delta, dict) else None
            else:
                delta = getattr(event, 'delta', None)
                text = getattr(delta, 'text', None)

            if text:
                text_chunks.append(text)
        return ''.join(text_chunks)

    @staticmethod
    def _parse_response(responses) -> str:
        """Extract text from a non-streaming Anthropic response."""
        for content in responses.content:
            if content.type == 'text':
                return content.text

        # If no text type content is found, return the first content
        # (backward compatibility).
        return responses.content[0].text

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

        with ThreadPoolExecutor() as executor:
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
        """Generate results given an input.

        Args:
            inputs (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, list, PromptList))

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
            system_prompts = []
        else:
            messages, system_prompts = self._convert_prompt_list(input)

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                api_params = {
                    'model': self.model,
                    'max_tokens': max_out_len,
                    'temperature': self.temperature,
                    'messages': messages,
                }
                if system_prompts:
                    api_params['system'] = '\n'.join(system_prompts)

                if self.thinking is not None:
                    api_params['thinking'] = self.thinking

                if self.stream:
                    api_params['stream'] = True

                if self.claude_extra_kwargs:
                    api_params.update(self.claude_extra_kwargs)

                responses = self.anthropic.messages.create(**api_params)

                if api_params.get('stream'):
                    return self._parse_stream_response(responses)
                return self._parse_response(responses)
            except Exception as e:
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError('Calling Claude API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')
