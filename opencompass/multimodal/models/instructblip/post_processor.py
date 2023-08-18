import re

import torch


class InstructBlipMMBenchPostProcessor:
    """"Post processor for MiniGPT-4 on MMBench."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer) -> str:

        output_text = tokenizer.decode(output_token,
                                       add_special_tokens=False)  # noqa
        output_text = [
            self._extract_key_words(text.strip()) for text in output_text
        ]
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
