import json
import os

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

from os import environ
from modelscope import MsDataset

@LOAD_DATASET.register_module()
class RaceDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = {}
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            for split in ['validation', 'test']:
                # 从 ModelScope 加载数据
                ms_dataset = MsDataset.load(path, subset_name=name, split=split)
                
                dataset_list = []
                for line in ms_dataset:
                    dataset_list.append({
                        'article': line['article'],
                        'question': line['question'],
                        'A': line['options'][0],
                        'B': line['options'][1],
                        'C': line['options'][2],
                        'D': line['options'][3],
                        'answer': line['answer'],
                    })
                dataset[split] = Dataset.from_list(dataset_list)
        else:
            for split in ['validation', 'test']:
                jsonl_path = os.path.join(path, split, f'{name}.jsonl')
                dataset_list = []
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = json.loads(line)
                        dataset_list.append({
                            'article': line['article'],
                            'question': line['question'],
                            'A': line['options'][0],
                            'B': line['options'][1],
                            'C': line['options'][2],
                            'D': line['options'][3],
                            'answer': line['answer'],
                        })
                dataset[split] = Dataset.from_list(dataset_list)
        return DatasetDict(dataset)
