import random
import re

import torch


class InstructBlipMMBenchPostProcessor:
    """"Post processor for MiniGPT-4 on MMBench."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:
        # convert output id 0 to 2 (eos_token_id)
        output_token[output_token == 0] = 2
        output_text = tokenizer.decode(output_token,
                                       add_special_tokens=False)  # noqa
        output_text = self._extract_key_words(output_text.strip())
        return output_text

    def _extract_key_words(self, output_text: str) -> str:

        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        output_text = output_text.strip('</s><s>')
        output_text = output_text.strip('</Img>')
        output_text = output_text.strip()
        pattern = re.compile(r'([A-Z]\.)')
        res = pattern.findall(output_text)
        if len(res) > 0:
            output_text = res[0][:-1]
        return output_text


class InstructBlipCOCOCaptionPostProcessor:
    """"Post processor for InstructBlip on COCO Caption."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:

        output_token[output_token == 0] = 2
        output_text = tokenizer.decode(output_token,
                                       add_special_tokens=False)  # noqa
        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        output_text = output_text.strip('</s><s>')
        output_text = output_text.strip('</Img>')
        output_text = output_text.strip()
        return output_text


class InstructBlipVQAPostProcessor:
    """"Post processor for InstructBlip on VQA."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:
        output_token[output_token == 0] = 2
        output_text = tokenizer.decode(output_token,
                                       add_special_tokens=False)  # noqa
        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        output_text = output_text.strip('</s><s>')
        output_text = output_text.strip('</Img>')
        output_text = output_text.strip()
        return output_text


class InstructBlipScienceQAPostProcessor:
    """"Post processor for InstructBlip on ScienceQA."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:

        output_token[output_token == 0] = 2
        output_text = tokenizer.decode(output_token,
                                       add_special_tokens=False)  # noqa
        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        output_text = output_text.strip('</s><s>')
        output_text = output_text.strip('</Img>')
        output_text = output_text.strip()
        pattern = re.compile(r'\(([A-Z])\)')
        output_text = pattern.findall(output_text)
        if len(output_text) == 0:
            output_text = random.choice(['A', 'B', 'C', 'D'])
        else:
            output_text = output_text[0]
        return output_text


class InstructBlipVSRPostProcessor:
    """"Post processor for InstructBlip on VSR."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:

        output_token[output_token == 0] = 2
        output_text = tokenizer.decode(output_token, add_special_tokens=False)
        pattern = r'yes|no|Yes|No'
        output_text = re.findall(pattern, output_text)
        if len(output_text) > 0:
            output_text = output_text[0].lower()
        return output_text
