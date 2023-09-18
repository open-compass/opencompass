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


class QwenVLChatVSRPostProcessor:
    """VSR post processor for Qwen-VL-Chat."""

    def __init__(self) -> None:
        pass

    def __call__(self, response: str) -> str:
        if 'yes' in response.lower():
            return 'yes'
        elif 'no' in response.lower():
            return 'no'
        else:
            return 'unknown'
