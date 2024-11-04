# flake8: noqa
# yapf: disable

import json
import os.path as osp

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
        path = get_data_path(path, local_mode=False)
        dataset = DatasetDict()
        name = name.split('_')[-1]
        raw_data = []
        filename = osp.join(path, name, 'test.jsonl')
        with open(filename, encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f.readlines()]
        dataset['test'] = Dataset.from_list(raw_data)
        return dataset
