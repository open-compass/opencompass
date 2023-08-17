from typing import Any

import torch


class VisualGLMPostProcessor:
    """"Post processor for VisualGLM on MMBench."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor, tokenizer: Any,
                 input_len: int) -> str:
        return tokenizer.decode(output_token[input_len:])
