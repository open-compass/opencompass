import json
import os.path as osp
import re

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)

from .base import BaseDataset


@TEXT_POSTPROCESSORS.register_module('charm-reason')
def charm_reason_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    match = re.search(r'\(([A-Z])\)*', ans)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])', ans)
    if match:
        return match.group(1)
    return ans


@ICL_EVALUATORS.register_module()
class CharmReasonEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        details = []
        cnt = 0
        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            if pred == ref:
                cnt += 1
                detail['correct'] = True
            details.append(detail)
        score = cnt / len(predictions) * 100
        return {'score': score, 'details': details}


@LOAD_DATASET.register_module()
class CharmDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        with open(osp.join(path, f'{name}.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)['examples']
        dataset = Dataset.from_list(data)
        return dataset
