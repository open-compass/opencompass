class OpenFlamingoVSRPostProcessor:
    """VSR post processor for Openflamingo."""

    def __init__(self) -> None:
        pass

    def __call__(self, raw_response: str) -> str:
        if 'yes' in raw_response.lower():
            return 'yes'
        elif 'no' in raw_response.lower():
            return 'no'
        else:
            return 'unknown'
