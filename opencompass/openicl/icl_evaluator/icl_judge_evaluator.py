# flake8: noqa
"""KOR-Bench Evaluator."""

import json
import os
import re

from .icl_base_evaluator import BaseEvaluator


class JudgeEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        correct = 0
        count = 0
        details = []
        for prediction, reference in zip(predictions, references):
            choice = prediction.split("\"Choice\": \"Model ")[-1][0]
            gold_winner = reference.get('winner', '')
            detail = {
                'pred': prediction,
                'answer': gold_winner,
                'correct': False
            }
            count += 1
            if choice == gold_winner:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result
