import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ChemBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_benchmark.json')
            with open(filename, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

            for item in data:
                raw_data.append({
                    'input': item['question'],
                    'A': item['A'],
                    'B': item['B'],
                    'C': item['C'],
                    'D': item['D'],
                    'target': item['answer'],
                })

            dataset[split] = Dataset.from_list(raw_data)
        return dataset
