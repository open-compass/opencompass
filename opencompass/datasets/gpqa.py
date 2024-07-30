import csv
import os
import random
import re

from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class GPQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path, local_mode=True)
        cnt = 0
        data = []
        with open(os.path.join(path, name), 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[7] == 'Question':
                    continue
                cnt = cnt + 1
                question = row[7]
                # 第一个是正确选项
                options = [row[8], row[9], row[10], row[11]]
                shuffle_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']  # 更新选项顺序
                c = shuffle_patterns[cnt % 4]
                line = {'question': question}
                ground_truth = options[0]
                for i in range(4):
                    line['ABCD'[i]] = options[ord(c[i]) - ord('A')]
                for i in range(4):
                    if line['ABCD'[i]] == ground_truth:
                        line['answer'] = 'ABCD'[i]
                        break
                data.append(line)
        dataset = Dataset.from_list(data)
        return dataset


class GPQAEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
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


@LOAD_DATASET.register_module()
class GPQASimpleEvalDataset(BaseDataset):
    """GPQA dataset compatible with simple-eval."""

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path, local_mode=True)
        n_repeats = 4
        data = []
        with open(os.path.join(path, name), 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[7] == 'Question':
                    continue
                question = row[7]
                # 第一个是正确选项
                options = [row[8], row[9], row[10], row[11]]
                line = {'question': question}
                line['answer'] = 'A'
                line['options'] = options
                data.append(line)

        data_list = data * n_repeats
        rng = random.Random(0)
        data_list = [
            data | {
                'permutation': rng.sample(range(4), 4)
            } for data in data_list
        ]
        for entry in data_list:
            options = entry['options']
            correct_options = [options[i] for i in entry['permutation']]
            for i in range(4):
                entry['ABCD'[i]] = correct_options[i]
            correct_index = entry['permutation'].index(0)
            correct_answer = 'ABCD'[correct_index]
            entry['options'] = correct_options
            entry['answer'] = correct_answer

        dataset = Dataset.from_list(data_list)
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def GPQA_Simple_Eval_postprocess(text: str) -> str:
    ANSWER_PATTERN = r'(?i)ANSWER\s*:\s*([A-D])'
    match = re.search(ANSWER_PATTERN, text)
    if match:
        return match.group(1)
    return None
