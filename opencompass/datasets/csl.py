import json

from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CslDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        dataset = load_dataset(**kwargs)

        def preprocess(example):
            keywords = '，'.join(example['keyword'])
            example['keywords'] = keywords

            return example

        dataset = dataset.map(preprocess)
        return dataset


@LOAD_DATASET.register_module()
class CslDataset_V2(BaseDataset):

    @staticmethod
    def load(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                item = {
                    'abst': line['abst'],
                    'keywords': '，'.join(line['keyword']),
                    'label': 'AB'[int(line['label'])],
                }
                data.append(item)
        return Dataset.from_list(data)
