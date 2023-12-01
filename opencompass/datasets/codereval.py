import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CoderEvalDataset(BaseDataset):

    @staticmethod
    def load(test_path):
        datasets = []
        with open(test_path, 'r', encoding='utf-8') as file:
            for line in file:
                dataset = json.loads(line.strip())
                datasets.append(dataset)
        return Dataset.from_list(datasets)
