# flake8: noqa
# yapf: disable

import json
import os

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MMMLUDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        subset = name.split('_')[1].replace('-', '_')
        for split in ['test']:
            data = load_dataset(path=path,
                                name=subset,
                                split=split,
                                trust_remote_code=True)
            dataset_list = []
            for item in data:
                dataset_list.append({
                    'input': item['Question'],
                    'A': item['A'],
                    'B': item['B'],
                    'C': item['C'],
                    'D': item['D'],
                    'target': item['Answer'],
                    'subject': item['Subject'].replace('_', ' ')
                })
            dataset[split] = Dataset.from_list(dataset_list)
        return dataset

@LOAD_DATASET.register_module()
class MMMLULiteDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        path = os.path.join(path, name + '.jsonl')
        dataset_list = []
        with open(path, 'r') as f:
            dataset_list = [json.loads(line) for line in f.readlines()]
        dataset['test'] = Dataset.from_list(dataset_list)
        return dataset
