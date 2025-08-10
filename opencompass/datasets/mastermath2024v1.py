import csv
import os

from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MastermathDatasetv1(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        cnt = 0
        data = []
        with open(os.path.join(path, name), 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[1] == 'question':
                    continue
                cnt = cnt + 1
                question = row[1]
                A = row[2]
                B = row[3]
                C = row[4]
                D = row[5]
                answer = row[6]
                data.append({
                    'question': question,
                    'A': A,
                    'B': B,
                    'C': C,
                    'D': D,
                    'answer': answer,
                })

        dataset = Dataset.from_list(data)

        return dataset


class MastermathDatasetv1Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if i == j:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result
