# flake8: noqa
# yapf: disable

import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MMMLUDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path, local_mode=True)
        dataset = DatasetDict()
        for split in ['test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    assert len(row) == 8
                    raw_data.append({
                        'input': row[1],
                        'A': row[2],
                        'B': row[3],
                        'C': row[4],
                        'D': row[5],
                        'target': row[6],
                        'subject':row[7].replace('_', ' '),
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset
