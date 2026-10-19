import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CMBDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        with open(osp.join(path, 'val.json'), 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        for d in val_data:
            d['option_str'] = '\n'.join(
                [f'{k}. {v}' for k, v in d['option'].items() if len(v) > 1])
        val_dataset = Dataset.from_list(val_data)

        with open(osp.join(path, 'test.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        for d in test_data:
            d['option_str'] = '\n'.join(
                [f'{k}. {v}' for k, v in d['option'].items() if len(v) > 1])
            d['answer'] = 'NULL'
        test_dataset = Dataset.from_list(test_data)

        return DatasetDict({'val': val_dataset, 'test': test_dataset})
