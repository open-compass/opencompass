import json
import os.path as osp
import re
from os import environ

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class BBHDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(path, subset_name=name, split='test')
        else:
            with open(osp.join(path, f'{name}.json'), 'r') as f:
                data = json.load(f)['examples']
            dataset = Dataset.from_list(data)
        return dataset


@TEXT_POSTPROCESSORS.register_module('bbh-mcq')
def bbh_mcq_postprocess(text: str) -> str:
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


@TEXT_POSTPROCESSORS.register_module('bbh-freeform')
def bbh_freeform_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    ans = ans.split('\n')[0].strip()

    if ans.endswith('.'):
        ans = ans[:-1].strip()

    match = re.search(r'\*\*(.*?)\*\*', ans)
    if match:
        return match.group(1)

    return ans


@ICL_EVALUATORS.register_module()
class BBHEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        predictions = [bbh_freeform_postprocess(pred) for pred in predictions]

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


@ICL_EVALUATORS.register_module()
class BBHEvaluator_mcq(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
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
