import json

from datasets import Dataset

from .base import BaseDataset


class AnliDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['label'] = {'c': 'A', 'e': 'B', 'n': 'C'}[line['label']]
                dataset.append(line)
        return Dataset.from_list(dataset)
