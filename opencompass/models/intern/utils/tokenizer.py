import re
from typing import List, Union

import torch


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
