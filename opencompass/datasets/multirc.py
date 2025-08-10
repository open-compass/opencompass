import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MultiRCDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        with open(path, 'r', errors='ignore') as in_f:
            rows = []
            for line in in_f:
                sample = json.loads(line.strip())
                passage = sample['passage']
                text = passage['text']
                questions = passage['questions']
                for question_dict in questions:
                    question = question_dict['question']
                    answers = question_dict['answers']
                    for answer_dict in answers:
                        answer = answer_dict['text']
                        label = answer_dict['label']
                        rows.append({
                            'text': text,
                            'question': question,
                            'answer': answer,
                            'label': label
                        })
            dataset = Dataset.from_dict({
                'text': [row['text'] for row in rows],
                'question': [row['question'] for row in rows],
                'answer': [row['answer'] for row in rows],
                'label': [row['label'] for row in rows]
            })
            return dataset


@LOAD_DATASET.register_module()
class MultiRCDatasetV2(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        with open(path, 'r', errors='ignore') as in_f:
            rows = []
            for line in in_f:
                sample = json.loads(line.strip())
                text = sample['passage']['text']
                for question_dict in sample['passage']['questions']:
                    question = question_dict['question']
                    answers = question_dict['answers']
                    for answer in answers:
                        rows.append({
                            'text': text,
                            'question': question,
                            'answer': answer['text'],
                            'label': 'BA'[answer['label']]
                        })
            return Dataset.from_list(rows)
