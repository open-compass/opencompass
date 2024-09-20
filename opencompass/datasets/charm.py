import json
import os.path as osp
import re
from typing import List, Union

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator, LMEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import build_dataset_from_cfg, get_data_path

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


UNCERTAIN_LIST = ['不确定', '无法确定', '无法回答', '不知道', '不认识']


def charm_memory_eval(pred: str, ref: Union[str, List[str]]) -> str:

    for uncertain in UNCERTAIN_LIST:
        if uncertain in pred:
            return '[错误]'

    is_negative = False
    if isinstance(ref, str):
        if ref.startswith('[not]'):
            # 部分CHARM记忆题目的ref是"[not]xxx"
            # 即xxx是一个负例，pred中不应该出现xxx
            # 例如：https://github.com/opendatalab/CHARM/blob/v1.0/data/CHARM/memorization/Chinese_Movie_and_Music_Recommendation.json#L45
            is_negative = True

            ref = ref[5:]  # 去掉[not]，保留xxx
        references = [ref]
    else:
        references = ref  # 部分CHARM记忆题目的ref是List[str]
    assert isinstance(references, list)

    for r in references:
        if r in pred:  # pred中包含ref
            if is_negative:
                return '[错误]'
            else:
                return '[正确]'

    if is_negative:  # 已验证pred中不包含ref，且ref是负例，所以pred是正确的
        return '[正确]'
    else:
        return '[错误]'


class CharmMemoryEvaluator(LMEvaluator):
    """本Evaluator是基于规则评判CHARM记忆题目的回答是否正确,
    只用于Chinese_Movie_and_Music_Recommendation这一个任务的评判。
    由于CHARM其他的记忆任务需要使用LLM作为judge（使用LMEvaluator），因而整个eval使用的是SubjectiveEvalTask。
    因此，本Evaluator的输入输出与LMEvaluator一致。"""

    def __init__(self, prompt_template=None, *nargs, **kwargs):

        if prompt_template is None:
            prompt_template = dict(
                type='PromptTemplate',
                template=dict(
                    round=[dict(role='HUMAN', prompt='')]))  # useless

        super().__init__(prompt_template, *nargs, **kwargs)

    def score(self, predictions, references, **kwargs):

        assert isinstance(predictions, dict)  # single-model scoring
        references = [{} for _ in range(len(predictions[0]['model_preds']))
                      ] if references is None else references
        predictions = predictions['model_preds']

        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        eval_results = [
            charm_memory_eval(pred, ref)
            for pred, ref in zip(predictions, references)
        ]

        dataset = None
        if self.dataset_cfg:
            dataset = build_dataset_from_cfg(self.dataset_cfg)

        output = dict()
        for i in range(len(predictions)):
            if dataset is not None:
                question = ''
                for col in dataset.reader.input_columns:
                    question += dataset.reader['test'][col][i] + '\n'
            output[str(i)] = {
                'origin_prompt': [{
                    'role':
                    'HUMAN',
                    'prompt':
                    f"[Question]: {question}[Assistant's Answer]: {predictions[i]}"  # noqa
                }],
                'prediction':
                eval_results[i],
                'gold':
                references[i],
            }

        return output


@LOAD_DATASET.register_module()
class CharmDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path, local_mode=True)
        with open(osp.join(path, f'{name}.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)['examples']
        dataset = Dataset.from_list(data)
        return dataset
