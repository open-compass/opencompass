import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class JsonlDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        data = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return Dataset.from_list(data)
