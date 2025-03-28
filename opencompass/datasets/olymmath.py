from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class OlymMATHDataset(BaseDataset):

    @staticmethod
    def load(path: str, subset: str):
        dataset = load_dataset(path, subset)
        return dataset
