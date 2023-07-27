import json

from datasets import Dataset, DatasetDict

from .base import BaseDataset


class CrowspairsDataset_CN(BaseDataset):

    @staticmethod
    def load(path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)

        def preprocess(example):
            example['label'] = 'A'
            return example

        dataset = Dataset.from_list(data).map(preprocess)
        return DatasetDict({'test': dataset})
