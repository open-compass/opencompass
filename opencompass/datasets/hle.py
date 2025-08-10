from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class HLEDataset(BaseDataset):

    @staticmethod
    def load(path: str, category: str | None = None):
        dataset = load_dataset(path)
        ds = dataset['test'].filter(lambda x: x['image'] == '')
        if category:
            ds = ds.filter(lambda x: x['category'] == category)
        ds = ds.rename_column('question', 'problem')
        dataset['train'] = ds
        dataset['test'] = ds
        return dataset
