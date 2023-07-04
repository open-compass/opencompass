from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class HFDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        return load_dataset(**kwargs)
