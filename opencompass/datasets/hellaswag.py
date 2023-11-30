import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class hellaswagDataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    'ctx': data['query'].split(': ', 2)[-1],
                    'A': data['choices'][0],
                    'B': data['choices'][1],
                    'C': data['choices'][2],
                    'D': data['choices'][3],
                    'label': data['gold'],
                })
        dataset = Dataset.from_list(dataset)
        return dataset


@LOAD_DATASET.register_module()
class hellaswagDataset_V2(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    'ctx': data['query'].split(': ', 1)[-1],
                    'A': data['choices'][0],
                    'B': data['choices'][1],
                    'C': data['choices'][2],
                    'D': data['choices'][3],
                    'label': 'ABCD'[data['gold']],
                })
        dataset = Dataset.from_list(dataset)
        return dataset


@LOAD_DATASET.register_module()
class hellaswagDataset_V3(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    'query': data['query'],
                    'A': data['choices'][0],
                    'B': data['choices'][1],
                    'C': data['choices'][2],
                    'D': data['choices'][3],
                    'gold': data['gold'],
                })
        dataset = Dataset.from_list(dataset)
        return dataset
