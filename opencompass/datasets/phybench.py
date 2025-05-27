import os.path as osp
import json
from datasets import Dataset
from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

@LOAD_DATASET.register_module()
class PhyBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str = None, **kwargs):
        path = get_data_path(path)

        file_path = osp.join(path, 'PHYBench-fullques_v1.json')

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        inputs = [item['content'] for item in raw_data]
        targets = [item['answer'] for item in raw_data]

        return Dataset.from_dict({
            'input': inputs,
            'target': targets
        })
