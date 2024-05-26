import json
import os

from datasets import Dataset, DatasetDict

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
                continue_prompt = prompt.split('_')[1]
                data_item = {
                    'prompt': prompt,
                    'only_option1': line['option1'],
                    'only_option2': line['option2'],
                    'opt1': prompt.replace('_', line['option1']),
                    'opt2': prompt.replace('_', line['option2']),
                    'answer': line['answer'],
                    'cont': continue_prompt,
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
                continue_prompt = prompt.split('_')[1]
                answer = line['answer']
                answer = ' AB'[int(answer)] if answer != '' else 'NULL'
                data_item = {
                    'prompt': prompt,
                    'only_option1': line['option1'],
                    'only_option2': line['option2'],
                    'opt1': prompt.replace('_', line['option1']),
                    'opt2': prompt.replace('_', line['option2']),
                    'answer': answer,
                    'cont': continue_prompt,
                }
                dataset_list.append(data_item)
        dataset_list = Dataset.from_list(dataset_list)
        return dataset_list


@LOAD_DATASET.register_module()
class winograndeDataset_V3(BaseDataset):
    """Disconnect from Huggingface, winograndeDataset_V2."""

    @staticmethod
    def load(path):
        dataset_dict = DatasetDict()
        for split in ['train_xs', 'dev']:
            filename = os.path.join(path, f'{split}.jsonl')
            dataset_list = []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    prompt = line['sentence']
                    continue_prompt = prompt.split('_')[1]
                    answer = line['answer']
                    answer = ' AB'[int(answer)] if answer != '' else 'NULL'
                    data_item = {
                        'prompt': prompt,
                        'only_option1': line['option1'],
                        'only_option2': line['option2'],
                        'opt1': prompt.replace('_', line['option1']),
                        'opt2': prompt.replace('_', line['option2']),
                        'answer': answer,
                        'cont': continue_prompt,
                    }
                    dataset_list.append(data_item)
            dataset_dict[split] = Dataset.from_list(dataset_list)
        return dataset_dict
