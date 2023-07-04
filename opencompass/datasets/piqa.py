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
