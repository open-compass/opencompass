from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.registry import MODELS
from opencompass.utils import PromptList

from ..base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class Claude(BaseAPIModel):
    """Model wrapper around Claude API.

    Args:
        key (str): Authorization key.
        path (str): The model to be used. Defaults to claude-2.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
        self,
        key: str,
        path: str = 'claude-2',
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        try:
            from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
        except ImportError:
            raise ImportError('Import anthropic failed. Please install it '
                              'with "pip install anthropic" and try again.')

        self.anthropic = Anthropic(api_key=key)
        self.model = path
        self.human_prompt = HUMAN_PROMPT
        self.ai_prompt = AI_PROMPT

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
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
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
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = f'{self.human_prompt} {input}{self.ai_prompt}'
        else:
            messages = ''
            for item in input:
                if item['role'] == 'HUMAN' or item['role'] == 'SYSTEM':
                    messages += f'{self.human_prompt} {item["prompt"]}'
                elif item['role'] == 'BOT':
                    messages += f'{self.ai_prompt} {item["prompt"]}'
            if not messages.endswith(self.ai_prompt):
                messages += self.ai_prompt

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                completion = self.anthropic.completions.create(
                    model=self.model,
                    max_tokens_to_sample=max_out_len,
                    prompt=messages)
                return completion.completion
            except Exception as e:
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError('Calling Claude API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')
