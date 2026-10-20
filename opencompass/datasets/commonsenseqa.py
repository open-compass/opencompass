import json
import os
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class commonsenseqaDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = {}
            for split in ['train', 'validation']:
                ms_dataset = MsDataset.load(path, split=split)
                dataset_list = []
                for line in ms_dataset:
                    choices = line['choices']
                    dataset_list.append({
                        'question': line['question'],
                        'A': choices['text'][0],
                        'B': choices['text'][1],
                        'C': choices['text'][2],
                        'D': choices['text'][3],
                        'E': choices['text'][4],
                        'answerKey': line['answerKey'],
                    })
                dataset[split] = Dataset.from_list(dataset_list)
        else:
            dataset = {}
            for split, stub in [
                ['train', 'train_rand_split.jsonl'],
                ['validation', 'dev_rand_split.jsonl'],
            ]:
                data_path = os.path.join(path, stub)
                dataset_list = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = json.loads(line)
                        dataset_list.append({
                            'question':
                            line['question']['stem'],
                            'A':
                            line['question']['choices'][0]['text'],
                            'B':
                            line['question']['choices'][1]['text'],
                            'C':
                            line['question']['choices'][2]['text'],
                            'D':
                            line['question']['choices'][3]['text'],
                            'E':
                            line['question']['choices'][4]['text'],
                            'answerKey':
                            line['answerKey'],
                        })
                dataset[split] = Dataset.from_list(dataset_list)
        return DatasetDict(dataset)
