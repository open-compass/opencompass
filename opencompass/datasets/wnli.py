from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class wnliDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        dataset = load_dataset(**kwargs)
        # dataset = dataset['validation']
        gt_dict = {
            1: 'A',
            0: 'B',
            -1: -1,
        }

        def preprocess(example):
            example['label_option'] = gt_dict[example['label']]
            return example

        return dataset.map(preprocess)
