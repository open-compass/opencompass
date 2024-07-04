import json
from os import environ

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class AFQMCDataset_V2(BaseDataset):

    @staticmethod
    def load(path):
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path, split='dev')
            data = []
            for line in ms_dataset:
                row = line
                row['label'] = 'AB'[int(line['label'])]
                data.append(line)
        else:
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    line['label'] = 'AB'[int(line['label'])]
                    data.append(line)
        return Dataset.from_list(data)
