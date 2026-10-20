import json
import os
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class PIQADataset(BaseDataset):

    @staticmethod
    def load_single(path, data_filename, label_filename):
        data_path = os.path.join(path, data_filename)
        label_path = os.path.join(path, label_filename)
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            data_lines = f.readlines()
        with open(label_path, 'r', encoding='utf-8') as f:
            label_lines = f.readlines()
        assert len(data_lines) == len(label_lines)
        for data, label in zip(data_lines, label_lines):
            i = json.loads(data.strip())
            i['label'] = int(label.strip())
            del i['id']
            dataset.append(i)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path)
            dataset = DatasetDict({
                'train': ms_dataset['train'],
                'validation': ms_dataset['validation']
            })
        else:
            train_dataset = PIQADataset.load_single(path, 'train.jsonl',
                                                    'train-labels.lst')
            val_dataset = PIQADataset.load_single(path, 'dev.jsonl',
                                                  'dev-labels.lst')
            dataset = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset
            })
        return dataset


@LOAD_DATASET.register_module()
class PIQADatasetV2(BaseDataset):

    @staticmethod
    def load_single(path, data_filename, label_filename):
        data_path = os.path.join(path, data_filename)
        label_path = os.path.join(path, label_filename)
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            data_lines = f.readlines()
        with open(label_path, 'r', encoding='utf-8') as f:
            label_lines = f.readlines()
        assert len(data_lines) == len(label_lines)
        for data, label in zip(data_lines, label_lines):
            i = json.loads(data.strip())
            label = int(label.strip())
            if label < 0:
                i['answer'] = 'NULL'
            else:
                i['answer'] = 'AB'[label]
            del i['id']
            dataset.append(i)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = DatasetDict()
            for split in ['train', 'validation']:
                ms_dataset = MsDataset.load(path, split=split)
                dataset_list = []
                for item in ms_dataset:
                    label = item['label']
                    dataset_list.append({
                        'goal':
                        item['goal'],
                        'sol1':
                        item['sol1'],
                        'sol2':
                        item['sol2'],
                        'answer':
                        'NULL' if label < 0 else 'AB'[label]
                    })
                dataset[split] = Dataset.from_list(dataset_list)
        else:
            train_dataset = PIQADatasetV2.load_single(path, 'train.jsonl',
                                                      'train-labels.lst')
            val_dataset = PIQADatasetV2.load_single(path, 'dev.jsonl',
                                                    'dev-labels.lst')
            dataset = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset
            })
        return dataset


@LOAD_DATASET.register_module()
class PIQADatasetV3(BaseDataset):

    @staticmethod
    def load_single(path, data_filename, label_filename):
        data_path = os.path.join(path, data_filename)
        label_path = os.path.join(path, label_filename)
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            data_lines = f.readlines()
        with open(label_path, 'r', encoding='utf-8') as f:
            label_lines = f.readlines()
        assert len(data_lines) == len(label_lines)
        for data, label in zip(data_lines, label_lines):
            i = json.loads(data.strip())
            i['label'] = int(label.strip())
            # some preprocessing
            i['goal'] = i['goal'][0].upper() + i['goal'][1:]
            if i['goal'].endswith('?') or i['goal'].endswith('.'):
                i['sol1'] = i['sol1'][0].upper() + i['sol1'][1:]
                i['sol2'] = i['sol2'][0].upper() + i['sol2'][1:]
            else:
                i['sol1'] = i['sol1'][0].lower() + i['sol1'][1:]
                i['sol2'] = i['sol2'][0].lower() + i['sol2'][1:]
            del i['id']
            dataset.append(i)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = DatasetDict()
            for split in ['train', 'validation']:
                ms_dataset = MsDataset.load(path, split=split)
                dataset_list = []
                for item in ms_dataset:
                    label = item['label']
                    goal = item['goal'][0].upper() + item['goal'][1:]
                    if goal.endswith('?') or goal.endswith('.'):
                        sol1 = item['sol1'][0].upper() + item['sol1'][1:]
                        sol2 = item['sol2'][0].upper() + item['sol2'][1:]
                    else:
                        sol1 = item['sol1'][0].lower() + item['sol1'][1:]
                        sol2 = item['sol2'][0].lower() + item['sol2'][1:]
                    dataset_list.append({
                        'goal': goal,
                        'sol1': sol1,
                        'sol2': sol2,
                        'label': label
                    })
                dataset[split] = Dataset.from_list(dataset_list)
        else:
            train_dataset = PIQADatasetV3.load_single(path, 'train.jsonl',
                                                      'train-labels.lst')
            val_dataset = PIQADatasetV3.load_single(path, 'dev.jsonl',
                                                    'dev-labels.lst')
            dataset = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset
            })
        return dataset
