import os.path as osp

from datasets import DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dev_dataset = load_dataset('csv',
                                   data_files=osp.join(path, 'dev',
                                                       f'{name}_dev.csv'),
                                   split='train')
        val_dataset = load_dataset('csv',
                                   data_files=osp.join(path, 'val',
                                                       f'{name}_val.csv'),
                                   split='train')
        val_dataset = val_dataset.add_column('explanation',
                                             [''] * len(val_dataset))
        test_dataset = load_dataset('csv',
                                    data_files=osp.join(
                                        path, 'test', f'{name}_test.csv'),
                                    split='train')
        test_dataset = test_dataset.add_column(
            'answer',
            [''] * len(test_dataset)).add_column('explanation',
                                                 [''] * len(test_dataset))
        return DatasetDict({
            'val': val_dataset,
            'dev': dev_dataset,
            'test': test_dataset
        })
