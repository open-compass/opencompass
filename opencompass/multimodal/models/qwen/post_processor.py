from typing import Any

import torch


class QwenVLBasePostProcessor:
    """Post processor for Qwen-VL-Base."""

    def __init__(self) -> None:
        pass

    def __call__(self, pred: torch.tensor, tokenizer: Any,
                 input_len: int) -> str:
        response = self.tokenizer.decode(pred)[input_len:]
        response = response.replace('<|endoftext|>', '').strip()
        return response
