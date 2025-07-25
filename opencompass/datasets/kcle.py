import json

import datasets

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class KCLEDataset(BaseDataset):

    @staticmethod
    def load(path, **kwargs) -> datasets.Dataset:
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['input'] = line['input']
                line['target'] = line['target']
                dataset.append(line)
        return datasets.Dataset.from_list(dataset)
