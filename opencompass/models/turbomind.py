import copy
from typing import Dict, List, Optional, Union

import numpy as np

from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

from .huggingface_above_v4_33 import _get_possible_max_seq_len

PromptType = Union[PromptList, str]


def valid_str(string, coding='utf-8'):
    """Decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


class TurboMindModel(BaseModel):
    """Model wrapper for TurboMind Python API.

    Args:
        path (str): path of the turbomind model
        backend (str): The infernce backend, which can be either 'turbomind' or
            'pytorch'. It will fallback to 'pytorch' once the model is not
            supported by 'turbomind'
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        engine_config (Dict, optional): The engine config to set
            arguments like session_len, max_batch_size for TurboMind.
        gen_config (Dict, optional): Generation config to set
                arguments like top_k, top_p, temperature.
        end_str (str, optional): Whether to trim generated strings with end_str
            if the model has special ending strings that are not handled well.
            Defaults to None.
    """

    def __init__(self,
                 path: str,
                 backend: str = 'turbomind',
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 engine_config: Dict = {},
                 gen_config: Dict = {},
                 batch_padding: bool = False,
                 drop_middle: bool = False,
                 end_str: Optional[str] = None):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)
        self.logger = get_logger()
        self.drop_middle = drop_middle
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        from lmdeploy import version_info
        from transformers import AutoTokenizer
        self.version_info = version_info
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)

        DEFAULT_ENGING_CONFIG = {'session_len': self.max_seq_len}
        _engine_config = DEFAULT_ENGING_CONFIG.copy()
        _engine_config.update(engine_config)
        self.pipe = self._build_pipe(path, backend, _engine_config)
        self.gen_config = gen_config
        self.batch_padding = batch_padding
        self.end_str = end_str

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
        assert isinstance(
            inputs, List), f'List(str) is expected, but got {type(inputs)}'

        stop_words = list(set(stopping_criteria))

        DEFAULT_GEN_CONFIG = {
            'max_new_tokens': max_out_len,
            'min_new_tokens': 1,
            'stop_words': stop_words,
        }

        gen_config = copy.deepcopy(DEFAULT_GEN_CONFIG)
        gen_config.update(self.gen_config)
        if do_sample:
            gen_config['top_k'] = 40
            gen_config['temperature'] = temperature
        else:
            if self.version_info >= (0, 6, 0):
                gen_config['do_sample'] = False
            else:
                gen_config['top_k'] = 1

        from lmdeploy import GenerationConfig
        gen_config = {
            k: v
            for k, v in gen_config.items() if hasattr(GenerationConfig, k)
        }
        gen_config = GenerationConfig(**gen_config)

        if self.drop_middle:
            inputs_drop_middle = []
            for input in inputs:
                input_ids = self.tokenizer([input],
                                           padding=False,
                                           truncation=False)['input_ids'][0]
                if len(input_ids) > self.max_seq_len:
                    input_ids = input_ids[:self.max_seq_len //
                                          2] + input_ids[-self.max_seq_len //
                                                         2:]
                    input = self.tokenizer.decode(input_ids,
                                                  skip_special_tokens=True)
                inputs_drop_middle.append(input)
            inputs = inputs_drop_middle

        results = []
        outputs = self.pipe(inputs, gen_config=gen_config, do_preprocess=False)
        for output in outputs:
            text = self.tokenizer.decode(output.token_ids)
            results.append(text)
        for s in stop_words:
            results = [r.split(s)[0] for r in results]
        return results

    def get_token_len(self, prompt: str) -> int:
        input_ids = self.tokenizer.encode(prompt)
        return len(input_ids)

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> np.ndarray:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            np.ndarray:  The perplexity scores in shape of (N,)
        """
        assert isinstance(
            inputs, List), f'List(str) is expected, but got {type(inputs)}'
        results = []
        if self.version_info <= (0, 6, 0):
            for text in inputs:
                input_ids = self.tokenizer.encode(text)
                res = self.pipe.get_ppl(input_ids)
                results.append(res)
            results = np.concatenate(results)
        else:
            if self.batch_padding and len(inputs) > 1:
                assert self.tokenizer.pad_token
                input_ids = self.tokenizer(
                    inputs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_len)['input_ids']
            else:
                input_ids = [
                    self.tokenizer(text)['input_ids'] for text in inputs
                ]
            for i in range(0, len(input_ids), 128):
                results.append(self.pipe.get_ppl(input_ids[i:i + 128]))
            results = np.concatenate(results)

        return results

    def get_loglikelihood(
            self,
            inputs: List[str],
            conts: List[str],
            mask_length: Optional[List[int]] = None) -> List[float]:
        assert isinstance(
            inputs, List), f'List(str) is expected, but got {type(inputs)}'
        results = []
        if self.version_info <= (0, 6, 0):
            for text, cont in zip(inputs, conts):
                input_ids = self.tokenizer.encode(text)
                res = self.pipe.get_ppl(input_ids)
                logit_sum = res * len(input_ids)
                input_ids = self.tokenizer.encode(text.replace(cont, ''))
                res = self.pipe.get_ppl(input_ids)
                logit_part = res * len(input_ids)
                results.append(-(logit_sum - logit_part))
            results = np.concatenate(results)
        else:
            for text, cont in zip(inputs, conts):
                input_ids = self.tokenizer.encode(text)
                res = self.pipe.get_ppl(input_ids)
                logit_sum = res * len(input_ids)
                input_ids = self.tokenizer.encode(text.replace(cont, ''))
                res = self.pipe.get_ppl(input_ids)
                logit_part = res * len(input_ids)
                results.append(-(logit_sum[0] - logit_part[0]))
            results = np.array(results)
        return results

    def _build_pipe(self, model_path, backend, engine_config):
        assert backend in ['pytorch', 'turbomind'], \
                f'unsupported backend type: {backend}'

        from lmdeploy import (PytorchEngineConfig, TurbomindEngineConfig,
                              pipeline)
        if backend == 'turbomind':
            filtered = {
                k: v
                for k, v in engine_config.items()
                if hasattr(TurbomindEngineConfig, k)
            }
            backend_config = TurbomindEngineConfig(**filtered)
        else:
            filtered = {
                k: v
                for k, v in engine_config.items()
                if hasattr(PytorchEngineConfig, k)
            }
            backend_config = PytorchEngineConfig(**filtered)
        return pipeline(model_path,
                        backend_config=backend_config,
                        log_level='INFO',
                        max_log_len=10)
