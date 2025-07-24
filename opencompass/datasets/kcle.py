import json
import datasets
from typing import List
from .base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator

@LOAD_DATASET.register_module()
class KCLEDataset(BaseDataset):

    @staticmethod
    def load(path, **kwargs) -> datasets.Dataset:
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['input'] = line['input']
                line['target'] = line['target']
                dataset.append(line)
        return datasets.Dataset.from_list(dataset)

class KCLEEvaluator(BaseEvaluator):

    def score(self, predictions: List, references: List) -> dict:
        pass

def kcle_postprocess(text: str) -> str:
    pass