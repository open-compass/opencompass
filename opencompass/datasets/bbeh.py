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
class BBEHDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(path, subset_name=name, split='test')
        else:
            with open(osp.join(path, f'{name}/task.json'), 'r') as f:
                data = json.load(f)['examples']
            dataset = Dataset.from_list(data)
        return dataset


@TEXT_POSTPROCESSORS.register_module('bbeh_freeform')
def bbeh_freeform_postprocess(text: str) -> str:
    # Extract answer using specified prefixes
    prefixes = [
        'The answer is: ', 'The answer is ', 'The final answer is: ',
        'The final answer is '
    ]
    answer = text
    for prefix in prefixes:
        if prefix in text:
            answer = text.split(prefix)[-1]
            break

    # Remove formatting markup
    if '\\boxed' in answer:
        answer = re.sub(r'\\boxed{(.*?)}', r'\1', answer)  # latex box
    if '\\text' in answer:
        answer = re.sub(r'\\text(?:tt)?{(.*?)}', r'\1', answer)  # text/texttt
    if '**' in answer:
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)  # bold

    # Take first line and clean
    if '\n' in answer:
        answer = answer.split('\n')[0].strip()

    return answer.strip().lower()


@TEXT_POSTPROCESSORS.register_module('bbeh_mcq')
def bbeh_mcq_postprocess(text: str) -> str:
    # Extract answer using specified prefixes
    prefixes = [
        'The answer is: ', 'The answer is ', 'The final answer is: ',
        'The final answer is '
    ]
    answer = text
    for prefix in prefixes:
        if prefix in text:
            answer = text.split(prefix)[-1]
            break

    # Remove parentheses if present
    answer = answer.strip('()')

    # Take first line and clean
    if '\n' in answer:
        answer = answer.split('\n')[0].strip()

    return answer.strip().lower()


@ICL_EVALUATORS.register_module()
class BBEHEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        processed_preds = [bbeh_freeform_postprocess(p) for p in predictions]
        # References are already in correct format
        processed_refs = [r.lower() for r in references]

        details = []
        correct_count = 0

        for pred, ref in zip(processed_preds, processed_refs):
            correct = False

            # Rule 1: Exact match
            if pred == ref:
                correct = True
            # Rule 2: Match after removing quotes/brackets
            elif pred == ref.strip("'\"()[]"):
                correct = True
            # Rule 4: Comma - separated answers
            elif ',' in ref:
                norm_pred = re.sub(r'\s*,\s*', ',', pred)
                norm_ref = re.sub(r'\s*,\s*', ',', ref)
                if norm_pred == norm_ref:
                    correct = True

            details.append({'pred': pred, 'answer': ref, 'correct': correct})
            correct_count += int(correct)

        score = (correct_count / len(predictions)) * 100
        return {'score': score, 'details': details}


@ICL_EVALUATORS.register_module()
class BBEHEvaluator_mcq(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        processed_preds = [bbeh_mcq_postprocess(p) for p in predictions]
        # References are already in correct format
        processed_refs = [r.lower().strip('()') for r in references]

        details = []
        correct_count = 0

        for pred, ref in zip(processed_preds, processed_refs):
            correct = False

            # Rule 1: Exact match
            if pred == ref:
                correct = True

            details.append({'pred': pred, 'answer': ref, 'correct': correct})
            correct_count += int(correct)

        score = (correct_count / len(predictions)) * 100
        return {'score': score, 'details': details}
