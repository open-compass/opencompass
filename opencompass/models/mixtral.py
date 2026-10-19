from typing import Dict, List, Optional, Union

import torch

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


class Mixtral(BaseModel):
    """Mixtral model wrapper https://github.com/open-compass/MixtralKit.

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
        max_batch_size: int = 8,
        tokenizer_only: bool = False,
        tokenizer_path: Optional[str] = None,
        meta_template: Optional[Dict] = None,
        num_gpus: int = 2,
    ):  # noqa
        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path)
        else:
            self._load_model(path=path,
                             max_seq_len=max_seq_len,
                             max_batch_size=max_batch_size,
                             tokenizer_path=tokenizer_path,
                             num_gpus=num_gpus)
        self.max_seq_len = max_seq_len
        self.template_parser = APITemplateParser(meta_template)
        self.logger = get_logger()

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    max_batch_size: int,
                    tokenizer_path: Optional[str] = None,
                    num_gpus: int = 2):
        from mixtralkit.mixtral import Mixtral
        self.generator = Mixtral.build(ckpt_dir=path,
                                       tokenizer_path=tokenizer_path,
                                       max_seq_len=max_seq_len,
                                       max_batch_size=max_batch_size,
                                       num_gpus=num_gpus)
        self.tokenizer = self.generator.tokenizer
        self.model = self.generator.model

    def _load_tokenizer(self, tokenizer_path: str):
        from mixtralkit.layers import Tokenizer
        self.tokenizer = Tokenizer(tokenizer_path)

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        prompt_tokens = []
        for input in inputs:
            tokens = self.tokenizer.encode(input, True, False)
            num_token = min(self.model.params.max_seq_len, len(tokens))
            prompt_tokens.append(tokens[-num_token:])
        generation_tokens, _ = self.generator.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_out_len,
            temperature=0,
        )
        results = [self.tokenizer.decode(t) for t in generation_tokens]
        return results

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        assert mask_length is None, 'mask_length is not supported'
        bsz = len(inputs)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        # tokenize
        prompt_tokens = [self.tokenizer.encode(x, True, False) for x in inputs]
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(params.max_seq_len, max_prompt_size)
        tokens = torch.zeros((bsz, total_len)).cuda().long()
        for k, t in enumerate(prompt_tokens):
            num_token = min(total_len, len(t))
            tokens[k, :num_token] = torch.tensor(t[-num_token:]).long()
        # forward
        outputs = self.model.forward(tokens, 0)
        # compute ppl
        shift_logits = outputs[..., :-1, :].contiguous().float()
        shift_labels = tokens[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        loss = loss_fct(shift_logits, shift_labels).view(bsz, -1)
        lens = (tokens != 0).sum(-1).cpu().numpy()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, True, True))
