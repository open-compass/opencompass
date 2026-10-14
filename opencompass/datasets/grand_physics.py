from datasets import load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class GrandPhysicsDataset(BaseDataset):

    @staticmethod
    def load(path: str, **kwargs):
        path = get_data_path(path)
        data = load_dataset('json', data_files=path, split='train', **kwargs)
        data = data.rename_columns({'problem': 'input', 'answer': 'target'})

        return data
