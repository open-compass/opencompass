from datasets import DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CivilCommentsDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        train_dataset = load_dataset(**kwargs, split='train')
        test_dataset = load_dataset(**kwargs, split='test')

        def pre_process(example):
            example['label'] = int(example['toxicity'] >= 0.5)
            example['choices'] = ['no', 'yes']
            return example

        def remove_columns(dataset):
            return dataset.remove_columns([
                'severe_toxicity', 'obscene', 'threat', 'insult',
                'identity_attack', 'sexual_explicit'
            ])

        train_dataset = remove_columns(train_dataset)
        test_dataset = remove_columns(test_dataset)
        test_dataset = test_dataset.shuffle(seed=42)
        test_dataset = test_dataset.select(list(range(10000)))
        test_dataset = test_dataset.map(pre_process)

        return DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
        })
