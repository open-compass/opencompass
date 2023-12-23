import json
import re
from pathlib import Path

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS


@LOAD_DATASET.register_module()
class CDMEDataset(BaseDataset):

    @staticmethod
    def load(path: str):

        data = {'prompt': [], 'answer': []}
        for file in Path(path).glob('*.jsonl'):
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line.strip())
                    data['prompt'].append(line['prompt'])
                    data['answer'].append(line['answer'])

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset


class CDMEEvaluator(BaseEvaluator):

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different lengths'
            }

        total_score = 0
        details = []
        for prediction, reference in zip(predictions, references):
            prediction = re.sub(r'\s+', '', prediction)
            reference = re.sub(r'\s+', '', reference)
            edit_distance = self.levenshtein_distance(prediction, reference)
            max_len = max(len(prediction), len(reference))
            score = 100 * (1 -
                           edit_distance / max_len) if max_len != 0 else 100

            detail = {
                'pred': prediction,
                'answer': reference,
                'edit_distance': edit_distance,
                'score': score
            }
            total_score += score
            details.append(detail)

        average_score = total_score / len(predictions) if predictions else 0
        result = {'score': average_score, 'details': details}
        return result


@TEXT_POSTPROCESSORS.register_module('cdme')
def cdme_postprocess(text: str) -> str:
    return text


@TEXT_POSTPROCESSORS.register_module('cdme_dataset')
def cdme_dataset_postprocess(text: str) -> str:
    return text
