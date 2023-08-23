from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class piqaDataset_V2(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            assert isinstance(example['label'], int)
            if example['label'] < 0:
                example['answer'] = 'NULL'
            else:
                example['answer'] = 'AB'[example['label']]
            example.pop('label')
            return example

        dataset = dataset.map(preprocess)
        return dataset


@LOAD_DATASET.register_module()
class piqaDataset_V3(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            example['goal'] = example['goal'][0].upper() + example['goal'][1:]
            if example['goal'].endswith('?') or example['goal'].endswith('.'):
                example['sol1'] = example['sol1'][0].upper(
                ) + example['sol1'][1:]
                example['sol2'] = example['sol2'][0].upper(
                ) + example['sol2'][1:]
            else:
                example['sol1'] = example['sol1'][0].lower(
                ) + example['sol1'][1:]
                example['sol2'] = example['sol2'][0].lower(
                ) + example['sol2'][1:]
            return example

        dataset = dataset.map(preprocess)
        return dataset
