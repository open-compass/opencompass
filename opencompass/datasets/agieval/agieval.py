import json
import os.path as osp
from os import environ

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .math_equivalence import is_equiv
from .post_process import parse_math_answer


@LOAD_DATASET.register_module()
class AGIEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, setting_name: str):
        path = get_data_path(path)
        from .dataset_loader import load_dataset, load_dataset_as_result_schema

        assert setting_name in 'zero-shot', 'only support zero-shot setting'
        dataset_wo_label = load_dataset(name, setting_name, path)
        dataset_with_label = load_dataset_as_result_schema(name, path)
        dataset = []
        for d1, d2 in zip(dataset_wo_label, dataset_with_label):
            dataset.append({
                'id': d2.index,
                'problem_input': d1['context'],
                'label': d2.label,
            })
        dataset = Dataset.from_list(dataset)
        return dataset


@LOAD_DATASET.register_module()
class AGIEvalDataset_v2(BaseDataset):

    @staticmethod
    def load(path: str, name: str, setting_name: str):
        path = get_data_path(path)
        assert setting_name in 'zero-shot', 'only support zero-shot setting'

        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path, subset_name=name, split='test')
            dataset = []
            for item in ms_dataset:
                passage = item['passage'] if item['passage'] else ''
                question = passage + item['question']
                options = '\n'.join(item['options']) if item['options'] else ''
                if item['label']:
                    try:
                        label = eval(item['label'])
                    except Exception:
                        label = item['label']
                    if isinstance(label, list):
                        label = ''.join(label)
                else:
                    label = item['answer']
                d = {'question': question, 'options': options, 'label': label}
                dataset.append(d)
            dataset = Dataset.from_list(dataset)
        else:
            filename = osp.join(path, name + '.jsonl')
            with open(filename, encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f]
            dataset = []
            for item in data:
                passage = item['passage'] if item['passage'] else ''
                question = passage + item['question']
                options = '\n'.join(item['options']) if item['options'] else ''
                if item['label']:
                    if isinstance(item['label'], list):
                        label = ''.join(item['label'])
                    else:
                        label = item['label']
                else:
                    label = item['answer']
                d = {'question': question, 'options': options, 'label': label}
                dataset.append(d)
            dataset = Dataset.from_list(dataset)
        return dataset


@ICL_EVALUATORS.register_module()
class AGIEvalEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        predictions = [parse_math_answer('', pred) for pred in predictions]
        details = []
        cnt = 0
        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            if is_equiv(pred, ref):
                cnt += 1
                detail['correct'] = True
            details.append(detail)
        score = cnt / len(predictions) * 100
        return {'score': score, 'details': details}


@ICL_EVALUATORS.register_module()
class AGIEvalEvaluator_mcq(BaseEvaluator):

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
