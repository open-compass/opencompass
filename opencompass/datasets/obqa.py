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


@LOAD_DATASET.register_module()
class OBQADataset_V2(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def pre_process(example):
            example['A'] = example['choices']['text'][0]
            example['B'] = example['choices']['text'][1]
            example['C'] = example['choices']['text'][2]
            example['D'] = example['choices']['text'][3]
            if not example['question_stem'].endswith('?'):
                example['question_stem'] += ' what?'
            return example

        dataset = dataset.map(pre_process).remove_columns(['id', 'choices'])
        return dataset
