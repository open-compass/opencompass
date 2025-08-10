from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch.distributed as dist

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


class LLaMA2AccessoryModel(BaseModel):
    """LLaMA2-Accessory model wrapper.

    Project: https://github.com/Alpha-VLLM/LLaMA2-Accessory

    Args:
        tokenizer_only (bool): whether to load tokenizer only
        meta_template (dict): meta template for the model
        additional_stop_symbols: (Iterable[str]): additional symbols that mark
            the end of generation, e.g. the "###" symbol for separating turns
            in the chat template.
        from_pretrained_kwargs: kwargs that will be passed to
            `accessory.MetaModel.from_pretrained` for model instantiation.
    """

    def __init__(self,
                 tokenizer_only: bool = False,
                 meta_template: Optional[Dict] = None,
                 additional_stop_symbols: Iterable[str] = (),
                 **from_pretrained_kwargs):
        if tokenizer_only:
            self._load_tokenizer(from_pretrained_kwargs)
        else:
            self._load_model(from_pretrained_kwargs)

        self.additional_stop_symbols = additional_stop_symbols
        self.max_seq_len = from_pretrained_kwargs.get('max_seq_len', 4096)
        self.template_parser = APITemplateParser(meta_template)
        self.logger = get_logger()

    def _load_model(self, from_pretrained_kwargs):
        from accessory.model.meta import MetaModel
        from accessory.util.misc import init_distributed_mode
        if not dist.is_initialized():
            init_distributed_mode()

        model_parallel_group = dist.GroupMember.WORLD
        from_pretrained_kwargs['mp_group'] = model_parallel_group

        self.model = MetaModel.from_pretrained(**from_pretrained_kwargs)
        self.tokenizer = self.model.tokenizer
        self.logger = get_logger()

    def _load_tokenizer(self, from_pretrained_kwargs):
        from accessory.model.tokenizer import (
            Tokenizer, probe_tokenizer_path_from_pretrained)
        if 'tokenizer_path' in from_pretrained_kwargs:
            tokenizer_path = from_pretrained_kwargs['tokenizer_path']
        else:
            pretrained_path = from_pretrained_kwargs['pretrained_path']
            if isinstance(pretrained_path, str):
                pretrained_path = [pretrained_path]
            tokenizer_path = probe_tokenizer_path_from_pretrained(
                pretrained_path[-1])

        self.tokenizer = Tokenizer(tokenizer_path)

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        results = self.model.generate(
            prompts=inputs,
            max_gen_len=max_out_len,
            temperature=0.,
            additional_stop_symbols=self.additional_stop_symbols)
        return results

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None):
        assert mask_length is None, 'mask_length is not supported'
        evaluation_results = self.model.evaluate_examples(examples=inputs)
        ppl = evaluation_results['ppl']
        return np.array(ppl, dtype=np.float32)

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, True, True))
