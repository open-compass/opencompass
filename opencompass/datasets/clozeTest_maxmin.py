import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MaxminDataset(BaseDataset):

    @staticmethod
    def load(test_path, answer_path=None):
        if answer_path is not None:
            with open(answer_path, 'r', encoding='utf-8') as answer_f:
                answers = {}
                for line in answer_f.readlines():
                    line = line.strip()
                    answers[line.split('<CODESPLIT>')[0]] = line.split(
                        '<CODESPLIT>')[1]
        datasets = []
        with open(test_path, 'r') as test_f:
            test_data = json.load(test_f)
            for item in test_data:
                dataset = dict()
                dataset['nl_tokens'] = ' '.join(item['nl_tokens'])
                dataset['pl_tokens'] = ' '.join(item['pl_tokens'])
                if answer_path is not None:
                    dataset['answer'] = 'A' if answers[
                        item['idx']] == 'max' else 'B'
                else:
                    dataset['answer'] = ''
                datasets.append(dataset)
        return Dataset.from_list(datasets)
