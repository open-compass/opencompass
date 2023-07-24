import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CMMLUDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                _ = next(reader)  # skip the header
                for row in reader:
                    assert len(row) == 7
                    raw_data.append({
                        'question': row[1],
                        'A': row[2],
                        'B': row[3],
                        'C': row[4],
                        'D': row[5],
                        'answer': row[6],
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset
