from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class commonsenseqaDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def pre_process(example):
            for i in range(5):
                example[chr(ord('A') + i)] = example['choices']['text'][i]
            return example

        dataset = dataset.map(pre_process).remove_columns(
            ['question_concept', 'id', 'choices'])
        return dataset
