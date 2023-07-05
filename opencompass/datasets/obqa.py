from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class OBQADataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def pre_process(example):
            for i in range(4):
                example[chr(ord('A') + i)] = example['choices']['text'][i]
            return example

        dataset = dataset.map(pre_process).remove_columns(['id', 'choices'])
        return dataset
