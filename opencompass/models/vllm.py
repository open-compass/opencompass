from typing import Dict, List, Optional

import numpy as np

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None

DEFAULT_MODEL_KWARGS = dict(trust_remote_code=True)


class VLLM(BaseModel):
    """Model Wrapper for VLLM."""

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        model_kwargs: dict = None,
        generation_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
        mode: str = 'none',
        use_fastchat_template: bool = False,
        stop_words: List[str] = [],
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)

        assert LLM, ('Please install VLLM with `pip install vllm`. '
                     'note: torch==2.1.2 is required.')
        self.logger = get_logger()
        self._load_model(path, model_kwargs)
        self.tokenizer = self.model.get_tokenizer()
        self.generation_kwargs = generation_kwargs
        self.generation_kwargs.pop('do_sample', None)

        assert mode in ['none', 'mid']
        self.mode = mode
        self.use_fastchat_template = use_fastchat_template
        self.stop_words = stop_words

    def _load_model(self,
                    path: str,
                    add_model_kwargs: dict = None,
                    num_retry: int = 3):
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        if add_model_kwargs is not None:
            model_kwargs.update(add_model_kwargs)
        import ray

        if ray.is_initialized():
            self.logger.info('shutdown ray instance to avoid '
                             '"Calling ray.init() again" error.')
            ray.shutdown()
        self.model = LLM(path, **model_kwargs)

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """

        if self.mode == 'mid':
            input_ids = self.tokenizer(inputs, truncation=False)['input_ids']
            inputs = []
            for input_id in input_ids:
                if len(input_id) > self.max_seq_len - max_out_len:
                    half = int((self.max_seq_len - max_out_len) / 2)
                    inputs.append(
                        self.tokenizer.decode(input_id[:half],
                                              skip_special_tokens=True) +
                        self.tokenizer.decode(input_id[-half:],
                                              skip_special_tokens=True))
                else:
                    inputs.append(
                        self.tokenizer.decode(input_id,
                                              skip_special_tokens=True))

        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)
        generation_kwargs.update({'max_tokens': max_out_len})
        _stop = list(set(self.stop_words + stopping_criteria))
        generation_kwargs.update({'stop': _stop})
        sampling_kwargs = SamplingParams(**generation_kwargs)
        outputs = self.model.generate(inputs, sampling_kwargs)

        prompt_list, output_strs = [], []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            prompt_list.append(prompt)
            output_strs.append(generated_text)

        return output_strs

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        batch_size = len(inputs)
        sampling_kwargs = SamplingParams(prompt_logprobs=0,
                                         **self.generation_kwargs)
        # forward
        outputs = self.model.generate(inputs, sampling_kwargs)
        # compute ppl
        ce_loss = []
        for i in range(batch_size):
            prompt_logprobs = outputs[i].prompt_logprobs[1:]
            prompt_token_ids = outputs[i].prompt_token_ids[1:]
            prompt_logprobs_list = [
                prompt_logprobs[i][prompt_token_ids[i]]
                for i in range(len(prompt_logprobs))
            ]
            prompt_logprobs_list = [i.logprob for i in prompt_logprobs_list]
            prompt_logprobs_list = np.array(prompt_logprobs_list)
            if mask_length is not None:
                prompt_logprobs_list = prompt_logprobs_list[-mask_length[i]:]
            loss = -prompt_logprobs_list.sum(axis=-1) / len(prompt_token_ids)
            ce_loss.append(loss)
        return np.array(ce_loss)

    def get_loglikelihood(self, inputs: List[str],
                          conts: List[str]) -> List[float]:
        mask_length = [
            self.get_token_len(c, add_special_tokens=False) for c in conts
        ]
        return -self.get_ppl(inputs, mask_length)

    def get_token_len(self,
                      prompt: str,
                      add_special_tokens: bool = True) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        tokenizer = self.model.get_tokenizer()
        token_ids = tokenizer.encode(prompt,
                                     add_special_tokens=add_special_tokens)
        return len(token_ids)
