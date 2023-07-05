from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class winograndeDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        dataset = load_dataset(**kwargs)

        def preprocess(example):
            prompt = example.pop('sentence')
            example['opt1'] = prompt.replace('_', example.pop('option1'))
            example['opt2'] = prompt.replace('_', example.pop('option2'))
            return example

        return dataset.map(preprocess)


@LOAD_DATASET.register_module()
class winograndeDataset_V2(BaseDataset):

    @staticmethod
    def load(**kwargs):

        dataset = load_dataset(**kwargs)

        def preprocess(example):
            prompt = example.pop('sentence')
            example['opt1'] = prompt.replace('_', example.pop('option1'))
            example['opt2'] = prompt.replace('_', example.pop('option2'))
            answer = example.pop('answer')
            if answer == '':
                example['label'] = 'NULL'
            else:
                example['label'] = ' AB'[int(answer)]
            return example

        return dataset.map(preprocess)
