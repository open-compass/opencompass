import re

from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CValuesDataset(BaseDataset):

    @staticmethod
    def load(path):

        dataset = load_dataset('json', data_files=path)

        def preprocess(example):
            example['prompt'] = re.sub('回复1', '回复A', example['prompt'])
            example['prompt'] = re.sub('回复2', '回复B', example['prompt'])
            example['label'] = re.sub('回复1', 'A', example['label'])
            example['label'] = re.sub('回复2', 'B', example['label'])
            return example

        return dataset.map(preprocess)
