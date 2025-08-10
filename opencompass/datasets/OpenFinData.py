import json
import os.path as osp

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class OpenFinDataDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path, local_mode=True)
        with open(osp.join(path, f'{name}.json'), 'r') as f:
            data = json.load(f)
            return Dataset.from_list(data)


@ICL_EVALUATORS.register_module()
class OpenFinDataKWEvaluator(BaseEvaluator):

    def __init__(self, ):
        super().__init__()

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        scores = []
        results = dict()

        for i in range(len(references)):
            all_hit = True
            judgement = references[i].split('、')
            for item in judgement:
                if item not in predictions[i]:
                    all_hit = False
                    break
            if all_hit:
                scores.append(True)
            else:
                scores.append(False)

        results['accuracy'] = round(sum(scores) / len(scores), 4) * 100
        return results
