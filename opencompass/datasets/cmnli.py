import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class cmnliDataset(BaseDataset):

    @staticmethod
    def load(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                if line['label'] == '-':
                    continue
                data.append(line)
        return Dataset.from_list(data)


@LOAD_DATASET.register_module()
class cmnliDataset_V2(BaseDataset):

    @staticmethod
    def load(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                if line['label'] == '-':
                    continue
                line['label'] = {
                    'entailment': 'A',
                    'contradiction': 'B',
                    'neutral': 'C',
                }[line['label']]
                data.append(line)
        return Dataset.from_list(data)
