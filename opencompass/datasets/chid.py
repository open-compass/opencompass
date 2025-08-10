import json

from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CHIDDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        if 'data_files' in kwargs:
            kwargs['data_files'] = get_data_path(kwargs['data_files'],
                                                 local_mode=True)
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            content = example['content']
            for i, c in enumerate(example['candidates']):
                example[f'content{i}'] = content.replace('#idiom#', c)
            return example

        dataset = dataset.map(preprocess)
        return dataset


@LOAD_DATASET.register_module()
class CHIDDatasetV2(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                item = {}
                item['content'] = line['content'].replace('#idiom#', '______')
                for i, c in enumerate(line['candidates']):
                    item[chr(ord('A') + i)] = c
                item['answer'] = 'ABCDEFG'[line['answer']]
                data.append(item)
        return Dataset.from_list(data)
