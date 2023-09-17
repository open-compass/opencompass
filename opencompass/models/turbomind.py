import os.path as osp
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.models.base import BaseModel
from opencompass.models.base_api import TokenBucket
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
from opencompass.models.base_api import APITemplateParser

from lmdeploy.serve.turbomind.chatbot import Chatbot
from lmdeploy.model import MODELS

import threading
import logging

PromptType = Union[PromptList, str]


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


class TurboMindModel(BaseModel):
    """Model wrapper for TurboMind API.

    Args:
        path (str): The name of OpenAI's model.
        model_path (str): folder of the turbomind model's path
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
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
        concurrency: int = 32,
        meta_template: Optional[Dict] = None,
    ):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)
        self.logger = get_logger()
        self.template_parser = APITemplateParser(meta_template)
        self.tis_addr = tis_addr
        self.concurrency = concurrency
        chatbot = Chatbot(self.tis_addr)
        self.model_name = chatbot.model_name
        self.chat_template = MODELS.get(self.model_name)()
        self.logger.warning(f'model_name: {self.model_name}')

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: float = 1.0,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
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
        results = []        
        dialogs = []
        for input in inputs:
            assert isinstance(input, (str, PromptList))
            if isinstance(input, str):
                dialog = [{'role': 'user', 'content': input}]
            else:
                dialog = []
                for item in input:
                    msg = {'content': item['prompt']}
                    if item['role'] == 'HUMAN':
                        msg['role'] = 'user'
                    elif item['role'] == 'BOT':
                        msg['role'] = 'assistant'
                    elif item['role'] == 'SYSTEM':
                        msg['role'] = 'system'
                    dialog.append(msg)
            dialogs.append(dialog)
            
            
        # chatbot = Chatbot(self.tis_addr, temperature=temperature, capability='completion', log_level=logging.ERROR)
        # tid = threading.currentThread().ident
        
        # for dialog in dialogs:
        #     prompt = self.chat_template.messages2prompt(dialog)
        #     for status, text, n_token in chatbot.stream_infer(
        #                 session_id=tid,
        #                 prompt=prompt,
        #                 request_output_len=max_out_len,
        #                 sequence_start=True, 
        #                 sequence_end=True):
        #             continue
       
    
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, dialogs,
                             [max_out_len] * len(dialogs),
                             [temperature] * len(dialogs)))
            # response = valid_str(text)
            # self.logger.error(f'****prompt: {prompt}\n\n****response: {response}')
            # results.append(text)
        return results

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

    def _generate(self, prompt: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
        """Generate results given a list of inputs.

        Args:
            prompt (str or PromptList): A string or PromptDict.
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
        # assert isinstance(prompt, (str, PromptList)), f'prompt type: {type(prompt)}'

        # assert type(
        #     prompt
        # ) is str, 'We only support string for TurboMind Python API now'
        chatbot = Chatbot(self.tis_addr, 
                          temperature=temperature, 
                          capability='completion',
                          top_k=1,
                          log_level=logging.ERROR)
        prompt = self.chat_template.messages2prompt(prompt)
        for status, text, n_token in chatbot.stream_infer(
                    session_id=threading.currentThread().ident,
                    prompt=prompt,
                    request_output_len=max_out_len,
                    sequence_start=True, 
                    sequence_end=True):
                continue
        response = valid_str(text)
        return response
