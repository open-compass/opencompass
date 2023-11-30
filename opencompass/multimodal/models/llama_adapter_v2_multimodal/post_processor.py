import torch


class LlamaAadapterMMBenchPostProcessor:
    """"Post processor for Llama Aadapter V2 on MMBench."""

    def __init__(self) -> None:
        pass

    def __call__(self, output_token: torch.tensor) -> str:

        if len(output_token) >= 2:
            if output_token[1] == '.':
                output_token = output_token[2:].strip()
        return output_token
