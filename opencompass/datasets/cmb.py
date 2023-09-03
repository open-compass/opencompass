import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CMBDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        with open(osp.join(path, 'test.json'), 'r') as f:
            test_data = json.load(f)
        with open(osp.join(path, 'val.json'), 'r') as f:
            val_data = json.load(f)

        for da in test_data:
            da['option_str'] = '\n'.join(
                [f'{k}. {v}' for k, v in da['option'].items() if len(v) > 1])
        for da in val_data:
            da['option_str'] = '\n'.join(
                [f'{k}. {v}' for k, v in da['option'].items() if len(v) > 1])

        test_dataset = Dataset.from_list(test_data)
        val_dataset = Dataset.from_list(val_data)
        return DatasetDict({'test': test_dataset, 'val': val_dataset})
