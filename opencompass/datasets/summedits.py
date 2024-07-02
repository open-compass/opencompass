import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset
from os import environ
from modelscope import MsDataset

@LOAD_DATASET.register_module()
class SummeditsDataset_V2(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = []
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            ms_dataset = MsDataset.load(path, split='train')
            for line in ms_dataset:
                row = line
                row['label'] = 'BA'[line['label']]
                row['original_summary'] = line['seed_summary']
                del row['seed_summary'], row['domain']
                dataset.append(row)
        else:
            with open(path, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    line['label'] = 'BA'[line['label']]
                    dataset.append(line)
        return Dataset.from_list(dataset)
