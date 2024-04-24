import csv
import os
import random
import re

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class GPQADataset_Simple_Eval(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
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


@TEXT_POSTPROCESSORS.register_module('first-capital-multi')
def GPQA_Simple_Eval_postprocess(text: str) -> str:
    ANSWER_PATTERN = r'(?i)ANSWER\s*:\s*([A-D])'
    match = re.search(ANSWER_PATTERN, text)
    if match:
        return match.group(1)
    return None
