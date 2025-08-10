import json

from datasets import Dataset, DatasetDict

from opencompass.utils import get_data_path

from .base import BaseDataset


class CrowspairsDatasetCN(BaseDataset):
    """Chinese version of Crowspairs dataset."""

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
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
