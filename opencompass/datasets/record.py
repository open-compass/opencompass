import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ReCoRDDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        with open(path, 'r', errors='ignore') as in_f:
            rows = []
            for i, line in enumerate(in_f):
                sample = json.loads(line.strip())
                passage = sample['passage']
                text = passage['text']
                text = text.replace('@highlight', '')

                qas = sample['qas']
                for qas_dict in qas:
                    query = qas_dict['query']
                    query = query.replace('@placeholder', '____')
                    answers = qas_dict['answers']
                    answers_temp = []
                    for answer_dict in answers:
                        answer = answer_dict['text']
                        answers_temp.append(answer)
                    rows.append({
                        'text': text,
                        'question': query,
                        'answers': answers_temp
                    })

            dataset = Dataset.from_dict({
                'text': [row['text'] for row in rows],
                'question': [row['question'] for row in rows],
                'answers': [row['answers'] for row in rows]
            })
            return dataset


class ReCoRDDatasetV2(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        with open(path, 'r', errors='ignore') as in_f:
            rows = []
            for i, line in enumerate(in_f):
                sample = json.loads(line.strip())
                text = sample['passage']['text'].replace('@highlight',
                                                         '').replace(
                                                             '\n\n', '\n')
                for qas_dict in sample['qas']:
                    query = qas_dict['query'].replace('@placeholder', '____')
                    answers = [
                        answer_dict['text']
                        for answer_dict in qas_dict['answers']
                    ]
                    rows.append({
                        'text': text,
                        'question': query,
                        'answers': answers
                    })

            dataset = Dataset.from_list(rows)
            return dataset


@TEXT_POSTPROCESSORS.register_module('ReCoRD')
def ReCoRD_postprocess(text: str) -> str:
    text = text.strip().split('\n')[0].replace('Answer: ', '').strip()
    return text
