# flake8: noqa

from opencompass.datasets.eese.utils import extract_first_numeric_score
from opencompass.registry import DICT_POSTPROCESSORS, TEXT_POSTPROCESSORS


@DICT_POSTPROCESSORS.register_module('eese_score_postprocess_dict')
def eese_score_postprocess_dict(output: dict, output_path: str) -> dict:
    """Post-process EESE score results for LLM judge - dict version"""
    # 处理每个预测结果
    for key, value in output.items():
        if 'prediction' in value:
            prediction = value['prediction']
            # 提取数字分数
            score = extract_first_numeric_score(prediction)
            if score is not None:
                value['score'] = score
            else:
                # 如果没有找到数字，尝试解析其他格式
                prediction_lower = prediction.strip().lower()
                if 'correct' in prediction_lower or 'right' in prediction_lower or '10' in prediction_lower:
                    value['score'] = 10
                elif 'incorrect' in prediction_lower or 'wrong' in prediction_lower or '0' in prediction_lower:
                    value['score'] = 0
                else:
                    value['score'] = 0  # 默认返回0分

    # 计算总体分数
    scores = [value.get('score', 0) for value in output.values()]
    if scores:
        overall_score = sum(scores) / (10 * len(scores))
    else:
        overall_score = 0

    return {'overall_score': overall_score, 'details': output}
