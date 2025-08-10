from datasets import concatenate_datasets, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class XCOPADataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        path = kwargs.get('path', None)
        lans = [
            'et', 'ht', 'it', 'id', 'qu', 'sw', 'zh', 'ta', 'th', 'tr', 'vi',
            'translation-et', 'translation-ht', 'translation-it',
            'translation-id', 'translation-sw', 'translation-zh',
            'translation-ta', 'translation-th', 'translation-tr',
            'translation-vi'
        ]

        datasets = []
        for lan in lans:
            dataset = load_dataset(path, lan)['validation']
            datasets.append(dataset)

        combined_dataset = concatenate_datasets(datasets)

        return combined_dataset
