import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MMLUCFDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        dataset = DatasetDict()
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            for split in ['dev', 'test']:
                # 从 ModelScope 加载数据
                if split == 'test':
                    _split = 'val'
                    ms_dataset = MsDataset.load(path,
                                            subset_name=name,
                                            split=_split)
                else:
                    ms_dataset = MsDataset.load(path,
                                            subset_name=name,
                                            split=split)
                
                dataset_list = []
                for i, line in ms_dataset:
                    if i == 0:  # 跳过第一行
                        continue
                    dataset_list.append({
                        'input': line['question'],
                        'A': line['choices'][0],
                        'B': line['choices'][1],
                        'C': line['choices'][2],
                        'D': line['choices'][3],
                        'target': 'ABCD'[line['answer']],
                    })
                dataset[split] = Dataset.from_list(dataset_list)
        else:
            for split in ['dev', 'test']:
                if split == 'test':
                    _split = 'val'
                    filename = osp.join(path, _split, f'{name}_{_split}.csv')
                else:
                    filename = osp.join(path, split, f'{name}_{split}.csv')
                raw_data = []
                with open(filename, encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  
                    for row in reader:

                        assert len(row) == 6
                        raw_data.append({
                            'input': row[0],
                            'A': row[1],
                            'B': row[2],
                            'C': row[3],
                            'D': row[4],
                            'target': row[5],
                        })
                dataset[split] = Dataset.from_list(raw_data)
        return dataset

