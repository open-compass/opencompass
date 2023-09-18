import random
import re

import torch


class OTTERMMBenchPostProcessor:
    """"Post processor for OTTER on MMBench."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:

        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = tokenizer.decode(output_token,
                                       add_special_tokens=False)  # noqa
        output_text = self._extract_key_words(output_text)
        return output_text

    def _extract_key_words(self, output_text: str) -> str:
        output_text = (output_text.split('<answer>')[-1].lstrip().rstrip().
                       split('<|endofchunk|>')[0].lstrip().rstrip())
        pattern = re.compile(r'([A-Z]\.)')
        res = pattern.findall(output_text)
        if len(res) > 0:
            output_text = res[0][:-1]
        return output_text


class OTTERCOCOCaptionPostProcessor:
    """"Post processor for OTTER on COCO Caption."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:

        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = tokenizer.decode(output_token,
                                       add_special_tokens=False)  # noqa
        output_text = (output_text.split('<answer>')[-1].lstrip().rstrip().
                       split('<|endofchunk|>')[0].lstrip().rstrip())
        pattern = re.compile(r'([A-Z]\.)')
        res = pattern.findall(output_text)
        if len(res) > 0:
            output_text = res[0][:-1]
        return output_text


class OTTERScienceQAPostProcessor:
    """"Post processor for OTTER on ScienceQA."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:

        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = tokenizer.decode(output_token,
                                       add_special_tokens=False)  # noqa
        output_text = (output_text.split('<answer>')[-1].lstrip().rstrip().
                       split('<|endofchunk|>')[0].lstrip().rstrip())
        pattern = re.compile(r'\(([A-Z])\)')
        output_text = pattern.findall(output_text)
        if len(output_text) == 0:
            output_text = random.choice(['A', 'B', 'C', 'D'])
        else:
            output_text = output_text[0]
        return output_text


class OTTERVQAPostProcessor:
    """"Post processor for OTTER on VQA."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:

        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = tokenizer.decode(output_token,
                                       add_special_tokens=False)  # noqa
        output_text = (output_text.split('<answer>')[-1].lstrip().rstrip().
                       split('<|endofchunk|>')[0].lstrip().rstrip())
        return output_text


class OTTERVSRPostProcessor:
    """"Post processor for OTTER on VSR."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:

        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = tokenizer.decode(output_token, add_special_tokens=False)
        pattern = r'yes|no|Yes|No'
        output_text = re.findall(pattern, output_text)
        if len(output_text) > 0:
            output_text = output_text[0].lower()
        return output_text


class OTTERMMEPostProcessor(OTTERMMBenchPostProcessor):
    """"Post processor for OTTER on MME."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:
        response = super().__call__(output_token, tokenizer)
        # extract yes or no, copy from MME official evaluation script
        prefix_pred_ans = response[:4].lower()

        if 'yes' in prefix_pred_ans:
            pred_label = 'yes'
        elif 'no' in prefix_pred_ans:
            pred_label = 'no'
        else:
            pred_label = 'other'

        return pred_label
