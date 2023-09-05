from typing import Dict, List, Optional

import numpy as np
import torch

from opencompass.models.base import BaseModel, LMTemplateParser


class InternLM(BaseModel):

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 tokenizer_path: Optional[str] = None,
                 model_config: Optional[str] = None,
                 tokenizer_type: Optional[str] = 'v7',
                 meta_template: Optional[Dict] = None):
        if tokenizer_only:
            self._load_tokenizer(tokenizer_path=tokenizer_path,
                                 tokenizer_type=tokenizer_type,
                                 max_seq_len=max_seq_len)
        else:
            self._load_model(path=path,
                             max_seq_len=max_seq_len,
                             tokenizer_path=tokenizer_path,
                             tokenizer_type=tokenizer_type,
                             model_config=model_config)
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    def _load_model(self,
                    path: str,
                    max_seq_len: int,
                    tokenizer_path: Optional[str] = None,
                    tokenizer_type: Optional[str] = None,
                    model_config: Optional[str] = None):

        from internlm.load.load_model import load_llm
        from internlm.model import build_model_with_cfg

        self.model, self.tokenizer, self.generator, _ = load_llm(
            path,
            max_seq_len,
            tokenizer_path=tokenizer_path,
            tokenizer_type=tokenizer_type,
            module=build_model_with_cfg,
            model_config_path=model_config)

    def _load_tokenizer(self, tokenizer_path: str, tokenizer_type: str,
                        max_seq_len: int):
        from internlm.load.tokenizer import LLMTokenizer
        from sentencepiece import SentencePieceProcessor
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        tokenizer = LLMTokenizer(tokenizer,
                                 max_seq_len=max_seq_len,
                                 tokenizer_type=tokenizer_type)
        self.tokenizer = tokenizer

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        tokens = self.tokenizer([prompt], truncation=False)['tokens']
        return len(tokens[0])

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        return self.generator.generate(inputs,
                                       generation_kwargs={
                                           'max_gen_len': max_out_len,
                                           'eos_token_id': self.eos_token_id
                                       })

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
        outputs, inputs = self.generator.get_logits(input_texts)

        shift_logits = outputs[..., :-1, :].contiguous().float()
        shift_labels = inputs['tokens'][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs['tokens'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss
