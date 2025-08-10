import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MMLUArabicDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path, local_mode=True)
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    assert len(row) == 6
                    raw_data.append({
                        'input': row[0],
                        'A': row[1],
                        'B': row[2],
                        'C': row[3],
                        'D': row[4],
                        'target': row[5],
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset
