import json

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class QuALITYDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        dataset_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                for question in line['questions']:
                    dataset_list.append({
                        'article':
                        line['article'],
                        'question':
                        question['question'],
                        'A':
                        question['options'][0],
                        'B':
                        question['options'][1],
                        'C':
                        question['options'][2],
                        'D':
                        question['options'][3],
                        'gold_label':
                        'ABCD'[question['gold_label'] - 1],
                        'difficult':
                        question['difficult']
                    })
        return Dataset.from_list(dataset_list)


class QuALITYEvaluator(BaseEvaluator):

    def score(self, predictions, references, test_set):
        assert len(predictions) == len(references)
        easy, hard, all = [], [], []
        for pred, refer, test in zip(predictions, references, test_set):
            if pred == refer:
                answer = True
            else:
                answer = False
            all.append(answer)
            if test['difficult'] == 0:
                easy.append(answer)
            else:
                hard.append(answer)
        return dict(easy_acc=sum(easy) / len(easy) * 100,
                    hard_acc=sum(hard) / len(easy) * 100,
                    all_acc=sum(all) / len(all) * 100)
