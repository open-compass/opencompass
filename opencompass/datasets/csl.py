import json

from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CslDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        if 'data_files' in kwargs:
            kwargs['data_files'] = get_data_path(kwargs['data_files'],
                                                 local_mode=True)
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            keywords = '，'.join(example['keyword'])
            example['keywords'] = keywords

            return example

        dataset = dataset.map(preprocess)
        return dataset


@LOAD_DATASET.register_module()
class CslDatasetV2(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
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
