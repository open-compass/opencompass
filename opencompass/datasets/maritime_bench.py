import json
import os.path as osp
from os import environ

import datasets
from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MaritimeBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str) -> datasets.Dataset:
        path = get_data_path(path)
        dataset = DatasetDict()
        dataset_list = []

        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            for split in ['test']:
                # 从 ModelScope 加载数据
                ms_dataset = MsDataset.load(path,
                                            subset_name=name,
                                            split=split)

                for line in ms_dataset:
                    question = line['question']
                    A = line['A']
                    B = line['B']
                    C = line['C']
                    D = line['D']
                    answer = line['answer']
                    dataset_list.append({
                        'question': question,
                        'A': A,
                        'B': B,
                        'C': C,
                        'D': D,
                        'answer': answer,
                    })
            # dataset[split] = Dataset.from_list(dataset_list)
        else:
            for split in ['test']:
                filename = osp.join(path, split, f'{name}_{split}.jsonl')
                with open(filename, encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        dataset_list.append({
                            'question': data['question'],
                            'A': data['A'],
                            'B': data['B'],
                            'C': data['C'],
                            'D': data['D'],
                            'answer': data['answer']
                        })

        dataset[split] = Dataset.from_list(dataset_list)

        return dataset
