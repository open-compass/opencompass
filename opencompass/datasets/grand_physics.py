import os

from datasets import load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class GrandPhysicsDataset(BaseDataset):

    @staticmethod
    def load(path: str, **kwargs):
        path = get_data_path(path)
        path = os.path.join(path)
        data = load_dataset(path)['train']
        data = data.rename_columns({'problem': 'input', 'answer': 'target'})

        return data
