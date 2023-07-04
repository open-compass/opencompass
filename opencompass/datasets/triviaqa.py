import csv
import os.path as osp
import re

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils.text_postprocessors import general_postprocess

from .base import BaseDataset


@LOAD_DATASET.register_module()
class TriviaQADataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            filename = osp.join(path, f'trivia-{split}.qa.csv')
            with open(filename) as f:
                reader = csv.reader(f, delimiter='\t')
                raw_data = []
                for row in reader:
                    assert len(row) == 2
                    question = row[0]
                    answers = eval(row[1])
                    if split == 'test':
                        answers = answers[0]
                    raw_data.append({'question': question, 'answer': answers})
                dataset[split] = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module()
class TriviaQAEvaluator(BaseEvaluator):

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        predictions = [
            re.split(r'[\n]', prediction, 1)[0].lower()
            for prediction in predictions
        ]
        processed_answers = [[general_postprocess(j).lower() for j in i]
                             for i in references]

        cnt = 0
        for pred, cand_ans in zip(predictions, processed_answers):
            cnt += int(any([cand in pred for cand in cand_ans]))
        score = cnt / len(predictions) * 100

        return {'score': score}
