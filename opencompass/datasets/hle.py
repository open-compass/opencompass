import json

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


@LOAD_DATASET.register_module()
class HLEVerifiedDataset(BaseDataset):

    @staticmethod
    def load(path: str,
             verified_classes: str | None = None,
             category: str | None = None):
        dataset = load_dataset(path)
        ds = dataset['train']
        ds = ds.filter(lambda x: json.loads(x['json'])['image'] == '')
        if verified_classes:
            ds = ds.filter(lambda x: x['Verified_Classes'] == verified_classes)
        if category:
            ds = ds.filter(lambda x: x['category'] == category)
        ds = ds.rename_column('question', 'problem')
        dataset['train'] = ds
        dataset['test'] = ds
        return dataset
