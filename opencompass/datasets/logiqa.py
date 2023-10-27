from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class LogiQADataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = load_dataset(path=path)

        def preprocess(example):
            options = example.pop('options')
            example['A'] = options[0]
            example['B'] = options[1]
            example['C'] = options[2]
            example['D'] = options[3]
            example['correct_option'] = chr(65 + example['correct_option'])
            return example

        dataset = dataset.map(preprocess)
        return dataset
