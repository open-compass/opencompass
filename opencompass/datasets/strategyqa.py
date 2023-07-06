import re

from opencompass.registry import TEXT_POSTPROCESSORS


@TEXT_POSTPROCESSORS.register_module('strategyqa')
def strategyqa_pred_postprocess(text: str) -> str:
    text = text.split('\n\n')[0]
    text = text.split('answer is ')[-1]
    match = re.search(r'(yes|no)', text.lower())
    if match:
        return match.group(1)
    return ''


@TEXT_POSTPROCESSORS.register_module('strategyqa_dataset')
def strategyqa_dataset_postprocess(text: str) -> str:
    return 'yes' if str(text) == 'True' else 'no'
