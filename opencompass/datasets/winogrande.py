import json
import os

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class winograndeDataset(BaseDataset):
    """Disconnect from Huggingface, winograndeDataset."""

    @staticmethod
    def load(path):
        path = os.path.join(path, 'dev.jsonl')
        dataset_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                prompt = line['sentence']
                continue_prompt = prompt.split('_')
                data_item = {
                    'opt1': prompt.replace('_', line['option1']),
                    'opt2': prompt.replace('_', line['option2']),
                    'answer': line['answer'],
                    'cont': continue_prompt[1]
                }
                dataset_list.append(data_item)
        dataset_list = Dataset.from_list(dataset_list)
        return dataset_list


@LOAD_DATASET.register_module()
class winograndeDataset_V2(BaseDataset):
    """Disconnect from Huggingface, winograndeDataset_V2."""

    @staticmethod
    def load(path):
        path = os.path.join(path, 'dev.jsonl')
        dataset_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                prompt = line['sentence']
                answer = line['answer']
                answer = ' AB'[int(answer)] if answer != '' else 'NULL'
                data_item = {
                    'opt1': prompt.replace('_', line['option1']),
                    'opt2': prompt.replace('_', line['option2']),
                    'answer': answer,
                }
                dataset_list.append(data_item)
        dataset_list = Dataset.from_list(dataset_list)
        return dataset_list
