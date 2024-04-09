import logging
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


class TurboMindTisModel(BaseModel):
    """Model wrapper for TurboMind Triton Inference Server gRPC API.

    Args:
        path (str): The name of OpenAI's model.
        tis_addr (str): The address (ip:port format) of turbomind's
            triton inference server
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
    """

    is_api: bool = True

    def __init__(
        self,
        path: str,
        tis_addr: str = '0.0.0.0:33337',
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)
        from lmdeploy.serve.turbomind.utils import Preprocessor
        self.preprocess = Preprocessor(tis_addr)
        self.logger = get_logger()
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']
        self.tis_addr = tis_addr

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

        Returns:
            List[str]: A list of generated strings.
        """

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results

    def get_token_len(self, prompt: str) -> int:
        input_ids, _ = self.preprocess(prompt)
        return input_ids.shape[-1]

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

    def _generate(self, prompt: PromptType, max_out_len: int,
                  temperature: float) -> str:
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

        from lmdeploy.serve.turbomind.chatbot import Chatbot
        chatbot = Chatbot(self.tis_addr,
                          temperature=temperature,
                          capability='completion',
                          top_k=1,
                          log_level=logging.ERROR)

        for status, text, n_token in chatbot.stream_infer(
                session_id=threading.currentThread().ident,
                prompt=prompt,
                request_output_len=max_out_len,
                sequence_start=True,
                sequence_end=True):
            continue
        response = valid_str(text)
        response = response.replace('<eoa>', '')
        return response
