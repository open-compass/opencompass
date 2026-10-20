from typing import Dict, List, Optional, Union

import torch

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


class Llama2(BaseModel):
    """LLaMA-2 model wrapper
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

    def get_loglikelihood(
            self,
            inputs: List[str],
            conts: List[str],
            mask_length: Optional[List[int]] = None) -> List[float]:
        assert mask_length is None, 'mask_length is not supported'
        bsz = len(inputs)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        # tokenize
        input_tokens = [self.tokenizer.encode(x, True, False) for x in inputs]
        max_prompt_size = max([len(t) for t in input_tokens])
        total_len = min(params.max_seq_len, max_prompt_size)
        tokens = torch.zeros((bsz, total_len)).cuda().long()
        num_token_list = []
        cont_tokens = []
        for k, t in enumerate(input_tokens):
            num_token = min(total_len, len(t))
            num_token_list.append(num_token - 1)
            tokens[k, :num_token] = torch.tensor(t[-num_token:]).long()
            context_ids = self.tokenizer.encode(
                inputs[k].replace(conts[k], ''), True, False)
            cont_tokens.append(tokens[k, len(context_ids):num_token])
        # forward
        outputs = self.model.forward(tokens, 0)[:, :-1, :]
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        loglikelihood_sum = torch.zeros(bsz).cuda()
        for idx in range(bsz):
            logits = outputs[
                idx, num_token_list[idx] -
                len(cont_tokens[idx]):num_token_list[idx], :].unsqueeze(0)
            loglikelihood_sum[idx] = torch.gather(
                logits, 2, cont_tokens[idx].unsqueeze(0).unsqueeze(-1)).sum()
        loglikelihood_sum = loglikelihood_sum.cpu().detach().numpy()
        return loglikelihood_sum

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, True, True))


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
        force_bf16 (bool): whether to force set model to `bfloat16`
    """

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        max_batch_size: int = 16,
        tokenizer_only: bool = False,
        tokenizer_path: Optional[str] = None,
        meta_template: Optional[Dict] = None,
        force_bf16: bool = False,
    ):  # noqa
        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path)
        else:
            self._load_model(path=path,
                             max_seq_len=max_seq_len,
                             max_batch_size=max_batch_size,
                             tokenizer_path=tokenizer_path,
                             force_bf16=force_bf16)
        self.max_seq_len = max_seq_len
        self.template_parser = APITemplateParser(meta_template)
        self.logger = get_logger()

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    max_batch_size: int,
                    tokenizer_path: Optional[str] = None,
                    force_bf16=False):
        from llama import Llama
        self.generator = Llama.build(path, tokenizer_path, max_seq_len,
                                     max_batch_size)
        self.tokenizer = self.generator.tokenizer
        self.model = self.generator.model
        if force_bf16:
            # force set model to `bfloat16` to fix
            # the exception of 'RuntimeError: probability tensor
            # contains either `inf`, `nan` or element < 0',
            # encountered during the inference of llama2-7b
            self.model = self.model.bfloat16()

    def _load_tokenizer(self, tokenizer_path: str):
        from llama import Tokenizer
        self.tokenizer = Tokenizer(tokenizer_path)

    def generate(self,
                 inputs: List[PromptType],
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
                    if item['role'].upper() == 'HUMAN':
                        msg['role'] = 'user'
                    elif item['role'].upper() == 'BOT':
                        msg['role'] = 'assistant'
                    elif item['role'].upper() == 'SYSTEM':
                        msg['role'] = 'system'
                    else:
                        raise ValueError(f'Unknown role: {item["role"]}')
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
            self.logger.warning('Batched data max token limit exceeded, '
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
