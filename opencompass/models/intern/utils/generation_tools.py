# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import re
from typing import List, Union

import torch

from opencompass.models.intern.utils.generation import _no_beam_search_generate

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


class LLMTokenizer(object):

    def __init__(self, tokenizer, max_seq_len=2048, tokenizer_type='llama'):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tokenizer_type = tokenizer_type
        if self.tokenizer_type == 'v4':
            self.bos_token_id = self.pad_token_id = 0
            self.eos_token_id = 1
        elif self.tokenizer_type in ['llama', 'v7']:
            self.bos_token_id = self.pad_token_id = 1
            self.eos_token_id = 2
        else:
            self.bos_token_id = self.pad_token_id = 1
            self.eos_token_id = 0

        # This is a hack to fit in with LLama type model
        self.bos_id = self.bos_token_id
        self.eos_id = self.eos_token_id
        self.pad_id = self.pad_token_id

    def __call__(self,
                 prompts,
                 padding=True,
                 right_align=False,
                 return_tensors='pt',
                 truncation=True):
        # import pdb; pdb.set_trace()
        if self.tokenizer_type == 'v4':
            tokens = [[0] + self.encode(x, False, False) for x in prompts]
        elif self.tokenizer_type in ['llama', 'v7']:
            tokens = [[1] + self.encode(x, False, False) for x in prompts]
        else:
            tokens = [self.encode(x, False, False) for x in prompts]

        if truncation:
            tokens = [i[:self.max_seq_len] for i in tokens]

        if padding:
            max_len = max([len(i) for i in tokens])
            if right_align:
                tokens = torch.LongTensor([[self.pad_token_id] *
                                           (max_len - len(i)) + i
                                           for i in tokens])
            else:
                tokens = torch.LongTensor([
                    i + [self.pad_token_id] * (max_len - len(i))
                    for i in tokens
                ])
        return {
            'tokens': tokens.cuda() if torch.cuda.is_available() else tokens
        }

    def encode(self, s: str, bos: bool, eos: bool):
        assert isinstance(s, str)
        s = self._process_meta_tokens(s)
        t = self._tokenize_list_str(s)
        if bos:
            t = [self.bos_token_id] + t
        if eos:
            t = t + [self.eos_token_id]
        return t

    def _process_meta_tokens(self, input_string: str) -> List[Union[str, int]]:
        # Create a pattern to match the META_TOKEN_{NUM} substrings
        pattern = re.compile(r'<META_TOKEN_(\d+)>')

        # Split the input string using the META_TOKEN_{NUM} substrings
        parts = pattern.split(input_string)

        # Combine the parts and tokens in the correct order
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text parts
                if part != '':
                    result.append(part)
            else:  # Meta token parts
                result.append(int(part))

        return result

    def _tokenize_list_str(self, s: Union[str, list]) -> List[int]:
        if isinstance(s, str):
            s = [s]
        assert isinstance(s, list)
        t = []
        for item in s:
            if isinstance(item, str):
                t += self.tokenizer.encode(item)
            elif isinstance(item, int):
                t.append(item)
            else:
                raise ValueError(f'Unsupported type {type(item)}')
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


class LLMGenerator:

    def __init__(self, model, tokenizer, use_mask=False, forward_kwargs=None):
        self.model = model
        self.tokenizer = tokenizer
        self.use_mask = use_mask
        self.forward_kwargs = forward_kwargs

    def generate(self,
                 inputs,
                 generation_kwargs={
                     'max_gen_len': 100,
                     'eos_token_id': None
                 }):
        tokenized_data = self.tokenizer(inputs,
                                        padding=True,
                                        right_align=True,
                                        return_tensors='pt')
        tokenized_data_len = tokenized_data['tokens'].shape[1]
        padding_data = self.tokenizer.tokenizer.decode(
            tokenized_data['tokens'].tolist())
        eos_token_id = generation_kwargs.get('eos_token_id')
        if not eos_token_id:
            eos_token_id = self.tokenizer.eos_token_id
        results = _no_beam_search_generate(
            self.model,
            tokenized_data['tokens'][..., ],
            do_sample=False,
            max_length=generation_kwargs['max_gen_len'] + tokenized_data_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            **self.forward_kwargs)
        results = results.squeeze(1).tolist()
        results_text = [
            self.tokenizer.tokenizer.decode(results[i])[len(padding_data[i]):]
            for i in range(len(inputs))
        ]

        def trunc_eos(text):
            eos_text = self.tokenizer.tokenizer.decode([eos_token_id])
            try:
                text = text[:text.index(eos_text)]
            except ValueError:
                pass
            return text

        if generation_kwargs.get('eos_token_id') is not None:
            results_text = [trunc_eos(t) for t in results_text]
        return results_text

    def get_logits(self, inputs):
        inputs = self.tokenizer(inputs,
                                padding=True,
                                return_tensors='pt',
                                truncation=True)
        if self.use_mask:
            outputs = self.model(input_ids=inputs['tokens'],
                                 **self.forward_kwargs)
        else:
            outputs = self.model(input_ids=inputs['tokens'])
        return outputs, inputs


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
