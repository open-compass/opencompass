import json
import os

from datasets import Dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from opencompass.datasets.matbench.post_process import (parse_float_answer,
                                                        parse_true_false_answer
                                                        )
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class MatbenchDataset(BaseDataset):

    @staticmethod
    def load(path, task):
        path = get_data_path(path)
        path = os.path.join(path,
                            'matbench_base_fold_0_' + task + '_test.json')
        dataset = []
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                dataset.append({
                    'problem': item['problem'],
                    'answer': item['answer'],
                })
        dataset = Dataset.from_list(dataset)
        return dataset


@ICL_EVALUATORS.register_module()
class MatbenchEvaluator_regression(BaseEvaluator):

    def score(self, predictions, references):
        mae_sum = 0
        count = 0
        details = []
        for pred, ref in zip(predictions, references):
            pred = parse_float_answer(pred)
            detail = {'pred': pred, 'answer': ref, 'error': None}
            count += 1
            try:
                error = abs(float(pred) - float(ref))
                mae_sum += error
                detail['error'] = error
            except Exception as e:
                detail['error'] = str(e)
            details.append(detail)
        mae = mae_sum / count if count > 0 else 0
        result = {'mae': mae, 'details': details}
        return result


@ICL_EVALUATORS.register_module()
class MatbenchEvaluator_classification(BaseEvaluator):

    def score(self, predictions, references):
        details = []
        predictions_parsed = []
        for pred, ref in zip(predictions, references):
            pred = parse_true_false_answer(pred)
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            if pred == ref:
                detail['correct'] = True
            details.append(detail)
            predictions_parsed.append(pred)
        accuracy = accuracy_score(references, predictions_parsed)
        precision = precision_score(references,
                                    predictions_parsed,
                                    average='binary')
        recall = recall_score(references, predictions_parsed, average='binary')
        f1 = f1_score(references, predictions_parsed, average='binary')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'details': details
        }
