import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SummeditsDataset_V2(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['label'] = 'BA'[line['label']]
                dataset.append(line)
        return Dataset.from_list(dataset)
