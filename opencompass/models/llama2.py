from typing import Dict, List, Optional, Union

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


class Llama2Chat(BaseModel):
    """LLaMA-2 chat model wrapper
    https://github.com/facebookresearch/llama/tree/main.

    Args:
        path (str): path to the model directory
        max_seq_len (int): max sequence length
        max_batch_size (int): max batch size
        tokenizer_only (bool): whether to load tokenizer only
        tokenizer_path (str): path to the tokenizer directory
        meta_template (dict): meta template for the model
    """

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        max_batch_size: int = 16,
        tokenizer_only: bool = False,
        tokenizer_path: Optional[str] = None,
        meta_template: Optional[Dict] = None,
    ):  # noqa
        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path)
        else:
            self._load_model(path=path,
                             max_seq_len=max_seq_len,
                             max_batch_size=max_batch_size,
                             tokenizer_path=tokenizer_path)
        self.max_seq_len = max_seq_len
        self.template_parser = APITemplateParser(meta_template)
        self.logger = get_logger()

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    max_batch_size: int,
                    tokenizer_path: Optional[str] = None):
        from llama import Llama
        self.generator = Llama.build(path, tokenizer_path, max_seq_len,
                                     max_batch_size)
        self.tokenizer = self.generator.tokenizer
        self.model = self.generator.model

    def _load_tokenizer(self, tokenizer_path: str):
        from llama import Tokenizer
        self.tokenizer = Tokenizer(tokenizer_path)

    def generate(self,
                 inputs: List[str or PromptList],
                 max_out_len: int = 512,
                 temperature: float = 0.6) -> str:
        """Generate response from input prompt.

        Args:
            inputs (list): input prompt
            max_out_len (int): max output length
            temperature (float): temperature for sampling
        """
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

        try:
            results = self.generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_out_len,
                temperature=temperature,
            )
            return [r['generation']['content'] for r in results]
        except AssertionError:
            self.warning('Batched data max token limit exceeded, '
                         'try to run one by one...')

        results = []
        for dialog in dialogs:
            try:
                result = self.generator.chat_completion(
                    [dialog],  # type: ignore
                    max_gen_len=max_out_len,
                    temperature=temperature,
                )[0]
                results.append(result['generation']['content'])
            except AssertionError:
                results.append('')
        return results

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, bos=True, eos=True)) + 100
