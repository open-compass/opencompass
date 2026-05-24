import re

from opencompass.registry import TEXT_POSTPROCESSORS


@TEXT_POSTPROCESSORS.register_module()
def arxivrollbench_selection_postprocess(text: str) -> str:
    """Extract an ArxivRollBench S/C answer in ``Selection N`` format."""
    if text is None:
        return ''
    text = str(text).strip()
    match = re.search(r'\bselection\s*([1-4])\b', text, re.IGNORECASE)
    if match:
        return f'Selection {match.group(1)}'
    match = re.search(r'\b([1-4])\b', text)
    if match:
        return f'Selection {match.group(1)}'
    return text


@TEXT_POSTPROCESSORS.register_module()
def arxivrollbench_choice_postprocess(text: str) -> str:
    """Extract an ArxivRollBench P-task answer in A/B/C/D format."""
    if text is None:
        return ''
    text = str(text).strip()
    match = re.search(r'\b([ABCD])\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return text
