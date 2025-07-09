from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class BeyondAIMEDataset(BaseDataset):

    @staticmethod
    def load(path, **kwargs):

        dataset = load_dataset(path=path, split='test')

        dataset = dataset.rename_column('problem', 'question')

        return dataset
