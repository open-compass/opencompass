import ast
import json
import math
import os
import re

from datasets import Dataset

from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class OpenSWIDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        new_data = []
        path = os.path.join(get_data_path(path), name)
        for file in os.listdir(path):
            if file.endswith('.jsonl'):
                final_path = os.path.join(path, file)
                with open(final_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        new_data.append({
                            'id': data['id_ddm'],
                            'prompt': data['dialogs'][0]['content'],
                            'ground_truth': data['ground_truth'],
                            'subset': file.split('.')[0]
                        })
        dataset = Dataset.from_list(new_data)
        return dataset


def extract_list(text):
    # 使用正则提取 ```python\n[ ... ]``` 中的 list 内容
    matches = re.findall(r'(\[.*?\])', text, re.DOTALL)
    if matches:
        raw_list_str = matches[-1]
        # 将字符串安全地转换为 Python 列表对象
        try:
            question_list = ast.literal_eval(raw_list_str)
            return question_list
        except Exception:
            return None
    else:
        return None


@ICL_EVALUATORS.register_module()
class OpenSWIMSEEvaluator(BaseEvaluator):
    """Exact match evaluator for name conversion."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        avg_score = 0
        avg_valid = []
        details = []
        for prediction, reference in zip(predictions, references):
            pred = extract_list(prediction)
            ans = reference

            if not pred or all(isinstance(x, float) for x in pred) is False:
                detail = {'pred': None, 'answer': ans, 'valid': False}
                pred = [0] * len(ans)
            else:
                detail = {'pred': pred, 'answer': ans, 'valid': True}
            if len(pred) < len(ans):
                detail['valid'] = False
                pred = pred + [0] * (len(ans) - len(pred))
            elif len(pred) > len(ans):
                detail['valid'] = False
                pred = pred[:len(ans)]
            avg_valid.append(detail['valid'])
            squared_errors = [(a - p)**2 for a, p in zip(ans, pred)]
            rmse_score = math.sqrt(sum(squared_errors) / len(squared_errors))
            detail['score'] = rmse_score
            avg_score += rmse_score
            details.append(detail)

        score = avg_score / len(predictions)
        valid = sum(avg_valid) / len(avg_valid)

        return {'score': score, 'valid': valid * 100, 'details': details}
