# flake8: noqa
# yapf: disable
import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

from .huggingface_above_v4_33 import (_convert_chat_messages,
                                      _format_with_fast_chat_template,
                                      _get_meta_template,
                                      _get_possible_max_seq_len)

PromptType = Union[PromptList, str]


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
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
        engine_config: Dict = {},
        gen_config: Dict = {},
        concurrency: int = 8,
        max_seq_len: int = None,
        meta_template: Optional[Dict] = None,
        fastchat_template: Optional[str] = None,
        stop_words: List[str] = [],
    ):
        from lmdeploy.messages import TurbomindEngineConfig
        from lmdeploy.turbomind import TurboMind
        from lmdeploy.version import version_info
        from transformers import AutoTokenizer

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)

        self.origin_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if not tokenizer_only:
            DEFAULT_ENGING_CONFIG = {'session_len': self.max_seq_len}
            _engine_config = DEFAULT_ENGING_CONFIG.copy()
            _engine_config.update(engine_config)
            engine_config = TurbomindEngineConfig(**_engine_config)
            tm_model = TurboMind.from_pretrained(path, engine_config=engine_config)
            self.tokenizer = tm_model.tokenizer
        self.generators = [tm_model.create_instance() for i in range(concurrency)]
        self.generator_ids = [i + 1 for i in range(concurrency)]
        self.concurrency = concurrency
        self.gen_config = gen_config
        self.version_info = version_info
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
                potential_stop_words.append(self.origin_tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.origin_tokenizer.decode(token_id))
        if self.origin_tokenizer.eos_token is not None:
            potential_stop_words.append(self.origin_tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

    def generate(self,
                 inputs: List[str],
                 max_out_len: int = 512,
                 stopping_criteria: List[str] = [],
                 do_sample: Optional[bool] = None,
                 temperature: int = 1,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of prompts
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        assert isinstance(inputs, List), f'List(str) is expected, but got {type(inputs)}'

        messages = _convert_chat_messages(inputs)
        if self.fastchat_template:
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
        else:
            messages = [self.origin_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]

        # split messages into batches
        batch_messages = [messages[i:i + self.concurrency] for i in range(0, len(messages), self.concurrency)]

        stop_words = list(set(self.stop_words + stopping_criteria))
        DEFAULT_GEN_CONFIG = {
            'max_new_tokens': max_out_len,
            'min_new_tokens': 1,
            'top_k': 1,
            'stop_words': stop_words,
        }
        gen_config = copy.deepcopy(DEFAULT_GEN_CONFIG)
        gen_config.update(self.gen_config)
        if do_sample:
            gen_config['top_k'] = 1000
            gen_config['temperature'] = temperature

        from lmdeploy.messages import EngineGenerationConfig, GenerationConfig
        gen_config = GenerationConfig(**gen_config)
        gen_config = EngineGenerationConfig.From(gen_config, self.tokenizer)

        results = []
        for batch_message in batch_messages:
            n = len(batch_message)
            with ThreadPoolExecutor() as executor:
                _results = list(
                    executor.map(
                        self._generate,
                        self.generators[:n],
                        self.generator_ids[:n],
                        batch_message,
                        [gen_config] * n,
                    ))
                results += _results

        for s in stop_words:
            results = [r.split(s)[0] for r in results]
        return results

    def _generate(self,
                  generator,
                  session_id,
                  prompt: PromptType,
                  gen_config=None) -> str:
        """Generate results given a list of inputs.

        Args:
            prompt (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            gen_config (EngineGenerationConfig, optional): Generation
                config to set arguments like top_k, top_p, temperature.
        Returns:
            str: The generated string.
        """
        assert type(prompt) is str, 'We only support string for TurboMind Python API'

        input_ids = self.tokenizer.encode(prompt, add_bos=False)
        for outputs in generator.stream_infer(session_id=session_id,
                                              input_ids=[input_ids],
                                              gen_config=gen_config,
                                              sequence_start=True,
                                              sequence_end=True,
                                              step=0,
                                              stream_output=False):
            if self.version_info >= (0, 4, 0):
                output_ids = outputs.token_ids
            else:
                _, output_ids, _ = outputs
            response = self.tokenizer.decode(output_ids)
            response = valid_str(response)
        return response

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        m = _convert_chat_messages([prompt])[0]
        t = self.origin_tokenizer.apply_chat_template(m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])
