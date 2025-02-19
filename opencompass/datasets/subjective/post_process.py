import re

from opencompass.registry import DICT_POSTPROCESSORS


@DICT_POSTPROCESSORS.register_module('think_pred')
def think_pred_postprocess(
    prediction: str,
    re_pattern: str,
) -> str:
    match = re.search(re_pattern, prediction)
    if match:
        return match.group(1).strip()
    else:
        return prediction
