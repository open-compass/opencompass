from typing import Any

import torch


class LLaVABasePostProcessor:
    """Base post processor for LLaVA on MMBench."""

    def __init__(self) -> None:
        pass

    def __call__(self, outputs: str, stop_str: str) -> str:
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        output_text = outputs.strip()
        return output_text


class LLaVAVSRPostProcessor(LLaVABasePostProcessor):
    """VSR post processor for LLaVA on MMBench."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, output_token: torch.tensor, tokenizer: Any,
                 input_len: int) -> str:
        output_text = tokenizer.decode(output_token[input_len:])
        if 'yes' in output_text.lower():
            return 'yes'
        elif 'no' in output_text.lower():
            return 'no'
        else:
            return 'unknown'
