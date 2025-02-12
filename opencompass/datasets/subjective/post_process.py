import re

from opencompass.registry import DICT_POSTPROCESSORS


@DICT_POSTPROCESSORS.register_module('infer_pred')
def infer_pred_postprocess(
    predictions: list,
    re_pattern: str,
) -> list:
    collect_list = []
    for prediction in predictions:
        match = re.search(re_pattern, prediction)
        if match:
            collect_list.append(match.group(1).strip())
        else:
            collect_list.append(prediction)
    return collect_list
