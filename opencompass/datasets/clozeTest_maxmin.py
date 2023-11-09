from datasets import Dataset
import json
from .base import BaseDataset

from opencompass.registry import LOAD_DATASET

@LOAD_DATASET.register_module()
class MaxminDataset(BaseDataset):

    @staticmethod
    def load(test_path, answer_path=None):
        if answer_path != None:
            with open(answer_path, 'r', encoding='utf-8') as answer_f:
                answers = {}
                for line in answer_f.readlines():
                    line = line.strip()
                    answers[line.split('<CODESPLIT>')[0]] = line.split('<CODESPLIT>')[1]
        datasets = []
        with open(test_path, 'r') as test_f:
            test_data = json.load(test_f)
            for item in test_data:
                dataset = dict()
                dataset['nl_tokens'] = " ".join(item['nl_tokens'])
                dataset['pl_tokens'] = " ".join(item['pl_tokens'])
                if answer_path != None:
                    dataset['answer'] = "A" if answers[item['idx']]=="max" else "B"
                else:
                    dataset['answer'] = ""
                datasets.append(dataset)
        return Dataset.from_list(datasets)
