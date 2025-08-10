from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class IWSLT2017Dataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)
        dataset = dataset.map(lambda example: example['translation']
                              ).remove_columns('translation')
        return dataset
