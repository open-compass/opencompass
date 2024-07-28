import json
import os
import re
from collections import Counter
from os import environ

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.utils import get_data_path
from opencompass.utils.text_postprocessors import general_postprocess

from .base import BaseDataset


class TydiQADataset(BaseDataset):

    @staticmethod
    def load(path, lang):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path, subset_name=lang, split='dev')
            dataset_list = []
            for line in ms_dataset:
                row = line
                answer = list(set([i['text'] for i in line['answers']]))
                row['answer'] = answer
                dataset_list.append(row)
        else:
            path = os.path.join(path, 'dev', f'{lang}-dev.jsonl')
            dataset_list = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    answer = list(set([i['text'] for i in line['answers']]))
                    line['answer'] = answer
                    dataset_list.append(line)
        return Dataset.from_list(dataset_list)


class TydiQAEvaluator(BaseEvaluator):
    # This evaluation class is edited from:
    #  https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
    def f1_score(self, prediction, ground_truth):
        prediction_tokens = general_postprocess(prediction).split()
        ground_truth_tokens = general_postprocess(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self, prediction, ground_truth):
        return (general_postprocess(prediction) == general_postprocess(
            ground_truth))

    def metric_max_over_ground_truths(self, metric_fn, prediction,
                                      ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def score(self, predictions, references):
        f1 = exact_match = total = 0
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        for prediction, reference in zip(predictions, references):
            prediction = re.split(r'[\n]', prediction, 1)[0].lower()
            exact_match += self.metric_max_over_ground_truths(
                self.exact_match_score, prediction, reference)
            f1 += self.metric_max_over_ground_truths(self.f1_score, prediction,
                                                     reference)
            total += 1

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        return {'exact_match': exact_match, 'f1': f1}
