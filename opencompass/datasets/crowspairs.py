from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class crowspairsDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        dataset = load_dataset(**kwargs)

        def preprocess(example):
            example['label'] = 0
            return example

        return dataset.map(preprocess)


@LOAD_DATASET.register_module()
class crowspairsDataset_V2(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            example['label'] = 'A'
            return example

        return dataset.map(preprocess)
