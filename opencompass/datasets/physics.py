import os

from datasets import load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class PHYSICSDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = os.path.join(path, name)
        path = get_data_path(path, local_mode=True)
        physics = load_dataset(path)['train']
        physics = physics.rename_column('questions', 'input')
        physics = physics.rename_column('final_answers', 'target')
        return physics
