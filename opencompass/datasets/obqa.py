import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class OBQADataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset_list = []
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
class OBQADataset_V2(BaseDataset):

    @staticmethod
    def load(path):
        dataset_list = []
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
