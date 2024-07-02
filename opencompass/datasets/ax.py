import json
from os import environ

from datasets import Dataset
from modelscope import MsDataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class AXDataset_V2(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = []
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            ms_dataset = MsDataset.load('opencompass/super_glue',
                                        subset_name='axb')['test']
            for data in ms_dataset:
                row = data
                row['label'] = {0: 'A', 1: 'B'}[data['label']]
                dataset.append(row)
        else:
            with open(path, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    line['label'] = {
                        'entailment': 'A',
                        'not_entailment': 'B'
                    }[line['label']]
                    dataset.append(line)
        dataset = Dataset.from_list(dataset)
        return dataset
