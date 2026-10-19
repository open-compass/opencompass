import json
import os
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class siqaDataset(BaseDataset):
    """Disconnect from HuggingFace version of HFDataset."""

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
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = DatasetDict()
            for split in ['train', 'validation']:
                data_list = []
                ms_dataset = MsDataset.load(path, split=split)
                for item in ms_dataset:
                    row = item
                    row['label'] = int(item['label'])
                    data_list.append(row)
                dataset[split] = Dataset.from_list(data_list)
            return dataset
        else:
            train_dataset = siqaDataset.load_single(path, 'train.jsonl',
                                                    'train-labels.lst')
            val_dataset = siqaDataset.load_single(path, 'dev.jsonl',
                                                  'dev-labels.lst')
            return DatasetDict({
                'train': train_dataset,
                'validation': val_dataset
            })


@LOAD_DATASET.register_module()
class siqaDataset_V2(BaseDataset):
    """Disconnect from HuggingFace version of siqaDataset_V2."""

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
            # some preprocessing
            i['all_labels'] = {
                'candidates': [
                    [f'A. {i["answerA"]}', 'A', i['answerA']],
                    [f'B. {i["answerB"]}', 'B', i['answerB']],
                    [f'C. {i["answerC"]}', 'C', i['answerC']],
                ],
                'label':
                label - 1
            }
            i['label'] = ' ABC'[label]

            dataset.append(i)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = DatasetDict()
            for split in ['train', 'validation']:
                data_list = []
                ms_dataset = MsDataset.load(path, split=split)
                for item in ms_dataset:
                    row = item
                    label = item['label']
                    # some preprocessing
                    row['all_labels'] = {
                        'candidates': [
                            [f'A. {item["answerA"]}', 'A', item['answerA']],
                            [f'B. {item["answerB"]}', 'B', item['answerB']],
                            [f'C. {item["answerC"]}', 'C', item['answerC']],
                        ],
                        'label':
                        int(label) - 1
                    }
                    row['label'] = ' ABC'[int(label)]

                    data_list.append(row)
                dataset[split] = Dataset.from_list(data_list)
        else:
            train_dataset = siqaDataset_V2.load_single(path, 'train.jsonl',
                                                       'train-labels.lst')
            val_dataset = siqaDataset_V2.load_single(path, 'dev.jsonl',
                                                     'dev-labels.lst')
            dataset = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset
            })
        return dataset


@LOAD_DATASET.register_module()
class SiqaDatasetV3(BaseDataset):
    """Disconnect from HuggingFace version of HFDataset."""

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
            i['A'] = i.pop('answerA')
            i['B'] = i.pop('answerB')
            i['C'] = i.pop('answerC')
            i['answer'] = 'ABC'[int(label.strip()) - 1]
            dataset.append(i)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = DatasetDict()
            for split in ['train', 'validation']:
                data_list = []
                ms_dataset = MsDataset.load(path, split=split)
                for item in ms_dataset:
                    row = item
                    label = item['label']
                    # some preprocessing
                    row['A'] = item['answerA']
                    row['B'] = item['answerB']
                    row['C'] = item['answerC']
                    row['answer'] = 'ABC'[int(label) - 1]
                    del row['answerA'], row['answerB'], row['answerC'], row[
                        'label']
                    data_list.append(row)
                dataset[split] = Dataset.from_list(data_list)
        else:
            train_dataset = SiqaDatasetV3.load_single(path, 'train.jsonl',
                                                      'train-labels.lst')
            val_dataset = SiqaDatasetV3.load_single(path, 'dev.jsonl',
                                                    'dev-labels.lst')
            dataset = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset
            })
        return dataset
