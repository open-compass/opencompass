import json
import os
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class StoryClozeDataset(BaseDataset):

    @staticmethod
    def load(path, lang):
        path = get_data_path(path)
        dataset_list = []
        for split in ['train', 'eval']:
            if environ.get('DATASET_SOURCE') == 'ModelScope':
                from modelscope import MsDataset
                ms_dataset = MsDataset.load(path,
                                            subset_name=lang,
                                            split=split)
                for line in ms_dataset:
                    line['context'] = ' '.join([
                        line['input_sentence_1'], line['input_sentence_2'],
                        line['input_sentence_3'], line['input_sentence_4']
                    ])
                    dataset_list.append(line)
            else:
                split_path = os.path.join(path, f'{lang}_{split}.jsonl')
                with open(split_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = json.loads(line)
                        line['context'] = ' '.join([
                            line['input_sentence_1'], line['input_sentence_2'],
                            line['input_sentence_3'], line['input_sentence_4']
                        ])
                        dataset_list.append(line)
        dataset_list = Dataset.from_list(dataset_list)
        return DatasetDict({'test': dataset_list})


@LOAD_DATASET.register_module()
class StoryClozeDatasetV2(BaseDataset):

    @staticmethod
    def load(path, lang):
        path = get_data_path(path)
        dataset_list = []
        for split in ['train', 'eval']:
            if environ.get('DATASET_SOURCE') == 'ModelScope':
                from modelscope import MsDataset
                ms_dataset = MsDataset.load(path,
                                            subset_name=lang,
                                            split=split)
                for line in ms_dataset:
                    line['context'] = ' '.join([
                        line['input_sentence_1'], line['input_sentence_2'],
                        line['input_sentence_3'], line['input_sentence_4']
                    ])
                    line['answer_right_ending'] = ' AB'[
                        line['answer_right_ending']]
                    dataset_list.append(line)
            else:
                split_path = os.path.join(path, f'{lang}_{split}.jsonl')
                with open(split_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = json.loads(line)
                        line['context'] = ' '.join([
                            line['input_sentence_1'], line['input_sentence_2'],
                            line['input_sentence_3'], line['input_sentence_4']
                        ])
                        line['answer_right_ending'] = ' AB'[
                            line['answer_right_ending']]
                        dataset_list.append(line)
        dataset_list = Dataset.from_list(dataset_list)
        return dataset_list
