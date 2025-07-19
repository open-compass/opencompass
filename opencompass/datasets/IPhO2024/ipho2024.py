import json

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class IPhO2024Dataset(BaseDataset):

    @staticmethod
    def load(path: str, **kwargs):
        path = get_data_path(path)
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        ipho = Dataset.from_list(data)

        return ipho
