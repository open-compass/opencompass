import json
from os import environ

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SummeditsDataset_V2(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path)
        dataset = []
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path, split='train')
            for line in ms_dataset:
                row = line
                row['label'] = 'BA'[line['label']]
                dataset.append(row)
        else:
            with open(path, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    line['label'] = 'BA'[line['label']]
                    dataset.append(line)
        return Dataset.from_list(dataset)
