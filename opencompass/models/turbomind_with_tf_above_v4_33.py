# flake8: noqa
# yapf: disable
import copy
import os
import time
from typing import Dict, List, Optional, Union

from mmengine.config.config import ConfigDict

from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

from .huggingface_above_v4_33 import (_convert_chat_messages,
                                      _format_with_fast_chat_template,
                                      _get_meta_template,
                                      _get_possible_max_seq_len)

PromptType = Union[PromptList, str]


def valid_str(string, coding='utf-8'):
    """Decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


class TurboMindModelwithChatTemplate(BaseModel):
    def __init__(
        self,
        path: str,
        tokenizer_only: bool = False,
        backend: str = 'turbomind',
        engine_config: Dict|ConfigDict = {},
        gen_config: Dict = {},
        max_seq_len: int = None,
        meta_template: Optional[Dict] = None,
        fastchat_template: Optional[str] = None,
        stop_words: List[str] = [],
        drop_middle: bool = False,
    ):
        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.drop_middle = drop_middle
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)

        from lmdeploy import version_info
        from transformers import AutoTokenizer
        self.version_info = version_info
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if not tokenizer_only:
            DEFAULT_ENGING_CONFIG = {'session_len': self.max_seq_len}
            _engine_config = DEFAULT_ENGING_CONFIG.copy()
            if isinstance(engine_config, ConfigDict):
                _engine_config.update(engine_config.to_dict())
            elif isinstance(engine_config, Dict):
                _engine_config.update(engine_config)
            else:
                raise ValueError(f'expected Dict or ConfigDict engine_config but got {type(engine_config)}')

            _engine_config.update(engine_config.to_dict())
            self.pipe = self._build_pipe(path, backend, _engine_config)
        else:
            self.pipe = None
        self.gen_config = gen_config
        self.fastchat_template = fastchat_template
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))
        self.logger.info(f'using stop words: {self.stop_words}')

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    stop_word = self.tokenizer.decode(token_id)
                    if stop_word.startswith(' '):
                        self.logger.warning(f'stop_word "{stop_word}" contains blanks, which will be stripped')
                        stop_word = stop_word.strip()
                    potential_stop_words.append(stop_word)
        if self.tokenizer.eos_token is not None:
            potential_stop_words.append(self.tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 do_sample: Optional[bool] = None,
                 temperature: float = 1.0,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of prompts
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.drop_middle:
            inputs_drop_middle = []
            for input in inputs:
                if isinstance(input, PromptList):
                    input = input[0]['prompt']
                input_ids = self.tokenizer([input],
                                           padding=False,
                                           truncation=False)['input_ids'][0]
                original_len = len(input_ids)
                # Reserve space for max_out_len in max_seq_len
                effective_max_len = self.max_seq_len - max_out_len
                if len(input_ids) > effective_max_len:
                    self.logger.info(f'Input length {original_len} exceeds effective sequence length {effective_max_len} (max_seq_len {self.max_seq_len} - max_out_len {max_out_len}), truncating...')
                    input_ids = input_ids[:effective_max_len //
                                          2] + input_ids[-effective_max_len //
                                                         2:]
                    self.logger.info(f'Input length after truncation: {len(input_ids)}')
                    input = self.tokenizer.decode(input_ids,
                                                  skip_special_tokens=True)
                inputs_drop_middle.append(input)
            inputs = inputs_drop_middle

        assert isinstance(inputs, List), f'List(str) is expected, but got {type(inputs)}'
        messages = _convert_chat_messages(inputs)
        if self.fastchat_template:
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
        else:
            # NOTE: DeepSeek-R1 series model's chat template will add <think> after the
            messages = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
            # LMDeploy tokenize prompts by AutoTokenizer with its default parameter "add_special_token=True"
            # OC add bos_token in the prompt, which requires tokenizing prompts using "add_speicial_token=False"
            # But LMDeploy doesn't have "add_speicial_token" in the pipeline API. So, we remove bos_token
            # from messages as a workaround
            if self.tokenizer.bos_token:
                bos_token = self.tokenizer.bos_token
                messages = [message.removeprefix(bos_token) if message.startswith(bos_token) else message for message in messages]
        stop_words = list(set(self.stop_words + stopping_criteria))

        DEFAULT_GEN_CONFIG = {
            'max_new_tokens': max_out_len,
            'min_new_tokens': 1,
            'stop_words': stop_words,
        }

        gen_config = copy.deepcopy(DEFAULT_GEN_CONFIG)
        gen_config.update(self.gen_config)
        if max_out_len is not None:
            gen_config['max_new_tokens'] = max_out_len
        if min_out_len is not None:
            gen_config['min_new_tokens'] = min_out_len
        if not(do_sample or ('do_sample' in self.gen_config and self.gen_config['do_sample'])):
            if self.version_info >= (0, 6, 0):
                gen_config['do_sample'] = False
            else:
                gen_config['top_k'] = 1

        from lmdeploy import GenerationConfig
        gen_config = {k: v for k, v in gen_config.items() if hasattr(GenerationConfig, k)}
        gen_config = GenerationConfig(**gen_config)
        self.logger.info('Generation Config of LMdeploy: ')
        self.logger.info(gen_config)

        results = []
        start = time.perf_counter()
        outputs = self.pipe(messages, gen_config=gen_config, do_preprocess=False)
        duration = time.perf_counter() - start
        input_tokens = [output.input_token_len for output in outputs]
        output_tokens = [output.generate_token_len for output in outputs]
        results = [output.text for output in outputs]
        self.logger.info(f'duration {duration:.2f}s, requests {len(inputs)}, input_tokens {sum(input_tokens)}, '
                         f'output_tokens {sum(output_tokens)}')

        for s in stop_words:
            results = [r.split(s)[0] for r in results]
        return results

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        m = _convert_chat_messages([prompt])[0]
        t = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])

    def _build_pipe(self, model_path, backend, engine_config):
        from lmdeploy import (PytorchEngineConfig, TurbomindEngineConfig,
                              pipeline)

        assert backend in ['pytorch', 'turbomind'], \
                f'unsupported backend type: {backend}'

        if backend == 'turbomind':
            filtered = {k: v for k, v in engine_config.items() if hasattr(TurbomindEngineConfig, k)}
            backend_config = TurbomindEngineConfig(**filtered)
        else:
            filtered = {k: v for k, v in engine_config.items() if hasattr(PytorchEngineConfig, k)}
            backend_config = PytorchEngineConfig(**filtered)

        log_level = os.getenv('LMDEPLOY_LOG_LEVEL', 'WARNING')
        max_log_len = os.getenv('LMDEPLOY_MAX_LOG_LEN', 10)
        return pipeline(model_path, backend_config=backend_config, log_level=log_level, max_log_len=max_log_len)
