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

    def __call__(self, outputs: str, stop_str: str) -> str:
        output_text = super().__call__(outputs, stop_str)
        if 'yes' in output_text.lower():
            return 'yes'
        elif 'no' in output_text.lower():
            return 'no'
        else:
            return 'unknown'
