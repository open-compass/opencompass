# flake8: noqa
# yapf: disable

import csv
import os.path as osp

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MMMLUDataset(BaseDataset):

    def transform(self, item):
        return {
            'input': item['Question'],
            'A': item['A'],
            'B': item['B'],
            'C': item['C'],
            'D': item['D'],
            'target': item['Answer'],
            'subject': item['Subject'].replace('_', ' ')
        }

    def load(self, path: str, name: str):
        subset = name.split('_')[1].replace('-', '_')
        data = load_dataset(path=path,
                            name='by_language',
                            trust_remote_code=True)[subset]
        dataset = DatasetDict()
        for split in ['test']:
            dataset[split] = data.map(self.transform)
        return dataset
