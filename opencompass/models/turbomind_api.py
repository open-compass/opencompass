import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
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
    """

    is_api: bool = True

    def __init__(self,
                 api_addr: str = 'http://0.0.0.0:23333',
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 end_str: Optional[str] = None,
                 **kwargs):
        super().__init__(path='',
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)
        from lmdeploy.serve.openai.api_client import APIClient
        self.chatbot = APIClient(api_addr)
        self.model_name = self.chatbot.available_models[0]
        self.logger = get_logger()
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']
        self.api_addr = api_addr
        self.end_str = end_str

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

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs),
                             [self.end_str] * len(inputs)))
        return results

    def get_token_len(self, prompt: str) -> int:
        input_ids, length = self.chatbot.encode(prompt)
        return length

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

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

        response = ''
        for output in self.chatbot.completions_v1(
                session_id=threading.currentThread().ident,
                prompt=prompt,
                model=self.model_name,
                max_tokens=max_out_len,
                temperature=temperature,
                top_p=0.8,
                top_k=1):
            response += output['choices'][0]['text']
        response = valid_str(response)
        if end_str:
            response = response.split(end_str)[0]
        return response
