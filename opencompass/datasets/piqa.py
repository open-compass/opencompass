import json
import os

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class piqaDataset(BaseDataset):

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
            dataset.append(i)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        train_dataset = piqaDataset.load_single(path, 'train.jsonl',
                                                'train-labels.lst')
        val_dataset = piqaDataset.load_single(path, 'dev.jsonl',
                                              'dev-labels.lst')
        return DatasetDict({'train': train_dataset, 'validation': val_dataset})


@LOAD_DATASET.register_module()
class piqaDataset_V2(BaseDataset):

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
            dataset.append(i)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        train_dataset = piqaDataset_V2.load_single(path, 'train.jsonl',
                                                   'train-labels.lst')
        val_dataset = piqaDataset_V2.load_single(path, 'dev.jsonl',
                                                 'dev-labels.lst')
        return DatasetDict({'train': train_dataset, 'validation': val_dataset})


@LOAD_DATASET.register_module()
class piqaDataset_V3(BaseDataset):

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

            dataset.append(i)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        train_dataset = piqaDataset_V3.load_single(path, 'train.jsonl',
                                                   'train-labels.lst')
        val_dataset = piqaDataset_V3.load_single(path, 'dev.jsonl',
                                                 'dev-labels.lst')
        return DatasetDict({'train': train_dataset, 'validation': val_dataset})
