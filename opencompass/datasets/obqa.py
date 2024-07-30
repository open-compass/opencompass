import json
from os import environ

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class OBQADataset(BaseDataset):

    @staticmethod
    def load(path, name='main'):
        path = get_data_path(path)
        dataset_list = []
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path, subset_name=name, split='test')
            for line in ms_dataset:
                item = {
                    'A': line['choices']['text'][0],
                    'B': line['choices']['text'][1],
                    'C': line['choices']['text'][2],
                    'D': line['choices']['text'][3],
                    'question_stem': line['question_stem'],
                    'answerKey': line['answerKey'],
                }
                if 'fact1' in line:
                    item['fact1'] = line['fact1']
                dataset_list.append(item)
        else:
            with open(path, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    item = {
                        'A': line['question']['choices'][0]['text'],
                        'B': line['question']['choices'][1]['text'],
                        'C': line['question']['choices'][2]['text'],
                        'D': line['question']['choices'][3]['text'],
                        'question_stem': line['question']['stem'],
                        'answerKey': line['answerKey'],
                    }
                    if 'fact1' in line:
                        item['fact1'] = line['fact1']
                    dataset_list.append(item)
        return Dataset.from_list(dataset_list)


@LOAD_DATASET.register_module()
class OBQADatasetV2(BaseDataset):

    @staticmethod
    def load(path, name='main'):
        path = get_data_path(path)
        dataset_list = []
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path, subset_name=name, split='test')
            for line in ms_dataset:
                question = line['question_stem']
                if not question.endswith('?'):
                    question += ' what?'
                item = {
                    'A': line['choices']['text'][0],
                    'B': line['choices']['text'][1],
                    'C': line['choices']['text'][2],
                    'D': line['choices']['text'][3],
                    'question_stem': question,
                    'answerKey': line['answerKey'],
                }
                if 'fact1' in line:
                    item['fact1'] = line['fact1']
                dataset_list.append(item)
        else:
            with open(path, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    question = line['question']['stem']
                    if not question.endswith('?'):
                        question += ' what?'
                    item = {
                        'A': line['question']['choices'][0]['text'],
                        'B': line['question']['choices'][1]['text'],
                        'C': line['question']['choices'][2]['text'],
                        'D': line['question']['choices'][3]['text'],
                        'question_stem': question,
                        'answerKey': line['answerKey'],
                    }
                    if 'fact1' in line:
                        item['fact1'] = line['fact1']
                    dataset_list.append(item)
        return Dataset.from_list(dataset_list)
