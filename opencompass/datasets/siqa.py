from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class siqaDataset_V2(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            example['all_labels'] = {
                'candidates': [
                    f'A. {example["answerA"]}',
                    f'B. {example["answerB"]}',
                    f'C. {example["answerC"]}',
                ],
                'label':
                int(example['label']) - 1
            }
            example['label'] = ' ABC'[int(example['label'])]
            return example

        dataset = dataset.map(preprocess)
        return dataset
