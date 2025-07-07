from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class HLEDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = load_dataset(path)
        dataset['test'] = dataset['test'].filter(lambda x: x['image'] == '')
        dataset['test'] = dataset['test'].rename_column('question', 'problem')
        dataset['train'] = dataset['test']
        return dataset
