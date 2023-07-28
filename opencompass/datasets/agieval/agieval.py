import json
import os.path as osp

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset
from .math_equivalence import is_equiv
from .post_process import parse_math_answer


@LOAD_DATASET.register_module()
class AGIEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, setting_name: str):
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
        assert setting_name in 'zero-shot', 'only support zero-shot setting'
        filename = osp.join(path, name + '.jsonl')
        with open(filename, encoding='utf-8') as f:
            _data = [json.loads(line.strip()) for line in f]
        data = []
        for _d in _data:
            passage = _d['passage'] if _d['passage'] else ''
            question = passage + _d['question']
            options = '\n'.join(_d['options']) if _d['options'] else ''
            label = _d['label'] if _d['label'] else _d['answer']
            d = {'question': question, 'options': options, 'label': label}
            data.append(d)
        dataset = Dataset.from_list(data)
        return dataset


@ICL_EVALUATORS.register_module()
class AGIEvalEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        predictions = [parse_math_answer('', pred) for pred in predictions]
        cnt = 0
        for pred, ref in zip(predictions, references):
            if is_equiv(pred, ref):
                cnt += 1
        score = cnt / len(predictions) * 100
        return {'score': score}
