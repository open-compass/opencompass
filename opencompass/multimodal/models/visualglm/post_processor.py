from typing import Any

import torch


class VisualGLMBasePostProcessor:
    """Base post processor for VisualGLM."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer: Any,
                 input_len: int) -> str:
        return tokenizer.decode(output_token[input_len:])


class VisualGLMVSRPostProcessor(VisualGLMBasePostProcessor):
    """VSR post processor for VisualGLM."""

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
