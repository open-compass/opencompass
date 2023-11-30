import json

from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class BoolQDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            if example['label'] == 'true':
                example['answer'] = 1
            else:
                example['answer'] = 0
            return example

        dataset = dataset.map(preprocess)
        return dataset


@LOAD_DATASET.register_module()
class BoolQDataset_V2(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['label'] = {'true': 'A', 'false': 'B'}[line['label']]
                dataset.append(line)
        return Dataset.from_list(dataset)


@LOAD_DATASET.register_module()
class BoolQDataset_V3(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['passage'] = ' -- '.join(
                    line['passage'].split(' -- ')[1:])
                line['question'] = line['question'][0].upper(
                ) + line['question'][1:]
                dataset.append(line)
        return Dataset.from_list(dataset)
