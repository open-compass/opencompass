import json
from typing import List

import datasets

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class KCLEDataset(BaseDataset):

    @staticmethod
    def load(path, **kwargs) -> datasets.Dataset:
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                data = {}
                data['input'] = line['input']
                data['target'] = line['target']
                dataset.append(data)
        return datasets.Dataset.from_list(dataset)


class KCLEEvaluator(BaseEvaluator):

    def score(self, predictions: List, references: List) -> dict:
        pass


def kcle_postprocess(text: str) -> str:
    pass
