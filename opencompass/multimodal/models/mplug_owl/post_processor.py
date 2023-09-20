import re

import torch


class MplugOwlMMBenchPostProcessor:
    """"Post processor for MplugOwl on MMBench."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor) -> str:
        pattern = re.compile(r'([A-Z]\.)')
        res = pattern.findall(output_token)
        if len(res) > 0:
            output_token = res[0][:-1]
        return output_token


class MplugOwlBasePostProcessor:
    """"Post processor for MplugOwl."""

    def __init__(self) -> None:
        pass

    def __call__(self, outputs: str) -> str:
        output_text = outputs.strip()
        return output_text
