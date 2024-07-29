import json
import os
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class WinograndeDataset(BaseDataset):
    """Disconnect from Huggingface, WinograndeDataset."""

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path,
                                        subset_name='winogrande_xs',
                                        split='validation',
                                        trust_remote_code=True)
            dataset_list = []
            for line in ms_dataset:
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
        else:
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
class WinograndeDatasetV2(BaseDataset):
    """Disconnect from Huggingface, WinograndeDatasetV2."""

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path,
                                        subset_name='winogrande_xs',
                                        split='validation',
                                        trust_remote_code=True)
            dataset_list = []
            for line in ms_dataset:
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
        else:
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
class WinograndeDatasetV3(BaseDataset):
    """Disconnect from Huggingface, WinograndeDatasetV2."""

    @staticmethod
    def load(path):
        path = get_data_path(path)
        dataset_dict = DatasetDict()
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            for split in ['train', 'validation']:
                ms_dataset = MsDataset.load(path,
                                            subset_name='winogrande_xs',
                                            split=split,
                                            trust_remote_code=True)
                dataset_list = []
                for line in ms_dataset:
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
                if split == 'train':
                    dataset_dict['train_xs'] = Dataset.from_list(dataset_list)
                elif split == 'validation':
                    dataset_dict['dev'] = Dataset.from_list(dataset_list)
        else:
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
