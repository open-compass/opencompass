from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class WMT19TranslationDataset(BaseDataset):

    @staticmethod
    def load(path: str, src_lang: str, tgt_lang: str, train_samples: int = 0):
        name = f'{src_lang}-{tgt_lang}'

        def prepare(split, limit=None):
            dataset = load_dataset(path,
                                   name=name,
                                   split=split,
                                   streaming=True)
            if limit is not None:
                dataset = dataset.take(limit)
            return Dataset.from_list([{
                'input': row['translation'][src_lang],
                'target': row['translation'][tgt_lang],
            } for row in dataset])

        splits = {'validation': prepare('validation')}
        if train_samples:
            splits['train'] = prepare('train', train_samples)
            if len(splits['train']) < 5:
                raise ValueError(f'Not enough training examples for {name}')
        return DatasetDict(splits)
