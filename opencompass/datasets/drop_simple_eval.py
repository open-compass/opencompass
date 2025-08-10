import json
import re
import string

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

# Modified from https://github.com/openai/simple-evals/blob/main/drop_eval.py

ANSWER_PATTERN = r'(?i)Answer\s*:\s*([^\n]+)'


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = ''.join(char for char in s if char not in exclude)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s


def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == '' or s2 == '':
        return s1 == s2

    return s1 in s2 or s2 in s1


@LOAD_DATASET.register_module()
class DropOpenAIDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        dataset_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                item = {
                    'prompt': data['context'],
                    'answers': data['ref_text'],
                }
                dataset_list.append(item)

        dataset_list = Dataset.from_list(dataset_list)
        return DatasetDict({'validation': dataset_list})


class DropOpenAIEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refers have different length'}
        num_correct = 0
        count = 0
        details = []
        for pred, refr in zip(predictions, references):
            match = re.search(ANSWER_PATTERN, pred)
            extracted_answer = match.group(1) if match else pred
            refrs = refr.split('|')
            matches = [
                fuzzy_match(extracted_answer, correct_answer)
                for correct_answer in refrs
            ]
            correct = True in matches
            num_correct += correct

            detail = {'pred': pred, 'answer': refr, 'correct': correct}
            count += 1

            details.append(detail)
        result = {'accuracy': 100 * num_correct / count, 'details': details}
        return result
