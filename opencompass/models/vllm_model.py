from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.registry import MODELS
from opencompass.utils import get_logger


@MODELS.register_module()
class VLLMModel(BaseModel):

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        tokenizer_only: bool = False,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: dict = dict(),
        model_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
        pad_token_id: Optional[int] = None,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            tokenizer_only=tokenizer_only,
            meta_template=meta_template,
        )
        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        self.stop = meta_template.get('stop', []) if meta_template else []

        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        if self.tokenizer.eos_token:
            self.stop.append(self.tokenizer.eos_token)

        if not tokenizer_only:
            self._load_model(
                path=path,
                model_kwargs=model_kwargs,
            )

        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    def _load_model(
        self,
        path: str,
        model_kwargs: dict,
    ):
        from vllm import LLM

        tensor_parallel_size = torch.cuda.device_count()
        self.logger.info(f'vllm tensor_parallel_size: {tensor_parallel_size}')
        self.model = LLM(
            model=path,
            tensor_parallel_size=tensor_parallel_size,
            **model_kwargs,
        )

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        # Copy from huggingface.py
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if self.pad_token_id is not None:
            if self.pad_token_id < 0:
                self.pad_token_id += self.tokenizer.vocab_size
            if self.tokenizer.pad_token_id is None:
                self.logger.warning(
                    f'Using {self.pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != self.pad_token_id:
                self.logger.warning(
                    f'pad_token_id is not consistent with the tokenizer. Using {self.pad_token_id} as pad_token_id'  # noqa
                )
            self.tokenizer.pad_token_id = self.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.logger.warning('pad_token_id is not set for the tokenizer.')
            if self.tokenizer.eos_token is not None:
                self.logger.warning('Using eos_token_id as pad_token_id.')
                self.logger.warning(
                    f'{self.tokenizer.eos_token} la {self.tokenizer.eos_token is None}'  # noqa
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError(
                    'pad_token_id is not set for this tokenizer. Try to set pad_token_id via passing `pad_token_id={PAD_TOKEN_ID}` in model_cfg. You may find pad_token_id in `generation.json`'  # noqa
                )

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=max_out_len,
                                         stop=self.stop)
        tokens = self.tokenizer(
            inputs,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_seq_len - max_out_len,
        )
        output = self.model.generate(
            prompt_token_ids=tokens['input_ids'],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return [it.outputs[0].text for it in output]

    def get_ppl(self,
                input_texts: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            input_texts (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out.

        Returns:
            List[float]: A list of perplexity scores.
        """
        from vllm import SamplingParams

        vocab_size = self.model.llm_engine.model_config.hf_config.vocab_size
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=vocab_size,
        )
        inputs = self.tokenizer(
            input_texts,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_seq_len - 1,
        )
        outputs = self.model.generate(
            prompt_token_ids=inputs['input_ids'],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        max_prompt_token_len = max(
            [len(it.prompt_token_ids) for it in outputs]) - 2
        shift_logprobs = []
        for output in outputs:
            _shift_logprobs = []
            # first token's prompt_logprobs is None
            for token_logprobs in output.prompt_logprobs[1:-1]:
                # TODO: too slow here
                tmp_tensor = torch.arange(vocab_size, dtype=torch.float)
                tmp_tensor.apply_(token_logprobs.get)
                _shift_logprobs.append(tmp_tensor.unsqueeze(0))
            for _ in range(max_prompt_token_len - len(_shift_logprobs)):
                _shift_logprobs.append(torch.zeros_like(_shift_logprobs[0]))

            _shift_logprobs = torch.cat(_shift_logprobs, dim=0)
            shift_logprobs.append(_shift_logprobs)
        shift_logprobs = torch.cat(shift_logprobs, dim=0)

        labels = pad_sequence(
            [torch.LongTensor(it) for it in inputs['input_ids']],
            batch_first=True,
            padding_value=-100,
        )
        shift_labels = labels[..., 2:].contiguous()

        loss_fct = torch.nn.NLLLoss(reduction='none', ignore_index=-100)

        loss = loss_fct(shift_logprobs.view(-1, shift_logprobs.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (shift_labels != -100).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss
