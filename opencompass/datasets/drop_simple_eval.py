import json
import re
import string
from typing import List, Set, Tuple

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

# Modified from https://github.com/openai/simple-evals/blob/main/drop_eval.py

ANSWER_PATTERN = r'(?i)Answer\s*:\s*([^\n]+)'


def _remove_articles(text: str) -> str:
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)


def _white_space_fix(text: str) -> str:
    return ' '.join(text.split())


EXCLUDE = set(string.punctuation)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _remove_punc(text: str) -> str:
    if _is_number(text):
        return text
    return ''.join(char for char in text if char not in EXCLUDE)


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    return text


def _normalize_answer(text: str) -> str:
    parts = [
        _white_space_fix(
            _remove_articles(_normalize_number(_remove_punc(token.lower()))))
        for token in re.split(r' |-', text)
    ]
    return ' '.join(part for part in parts if part.strip()).strip()


def _answer_to_bag(answer: str) -> Tuple[str, Set[str]]:
    normalized_answer = _normalize_answer(answer)
    return normalized_answer, set(normalized_answer.split())


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    precision = intersection / len(predicted_bag) if predicted_bag else 1.0
    recall = intersection / len(gold_bag) if gold_bag else 1.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 100 * (2 * precision * recall) / (precision + recall)


def _match_numbers_if_present(gold_bag: Set[str],
                              predicted_bag: Set[str]) -> bool:
    gold_numbers = {word for word in gold_bag if _is_number(word)}
    predicted_numbers = {word for word in predicted_bag if _is_number(word)}
    return not gold_numbers or bool(
        gold_numbers.intersection(predicted_numbers))


def get_drop_metrics(predicted: str, gold: str) -> Tuple[float, float]:
    """Return official DROP exact-match and F1 for single-span answers."""
    normalized_predicted, predicted_bag = _answer_to_bag(predicted)
    normalized_gold, gold_bag = _answer_to_bag(gold)
    exact_match = float(normalized_predicted == normalized_gold)
    f1 = 0.0
    if _match_numbers_if_present(gold_bag, predicted_bag):
        f1 = _compute_f1(predicted_bag, gold_bag)
    return exact_match, round(f1, 2)


def drop_metric(sample: str, references: List[str]) -> Tuple[float, float]:
    """Return the best DROP exact-match and F1 across valid references."""
    scores = [
        get_drop_metrics(sample, answer) for answer in references
        if answer.strip()
    ]
    em_scores, f1_scores = zip(*scores)
    return max(em_scores), max(f1_scores)


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
        exact_match = 0.0
        f1 = 0.0
        count = 0
        details = []
        for pred, refr in zip(predictions, references):
            match = re.search(ANSWER_PATTERN, pred)
            extracted_answer = match.group(1) if match else pred
            refrs = refr.split('|')
            em_score, f1_score = drop_metric(extracted_answer, refrs)
            matches = [
                fuzzy_match(extracted_answer, correct_answer)
                for correct_answer in refrs
            ]
            correct = True in matches
            num_correct += correct
            exact_match += em_score
            f1 += f1_score

            detail = {
                'pred': pred,
                'answer': refr,
                'correct': correct,
                'exact_match': em_score,
                'f1': f1_score,
            }
            count += 1

            details.append(detail)
        result = {
            'accuracy': 100 * num_correct / count,
            'exact_match': 100 * exact_match / count,
            'f1': f1 / count,
            'details': details,
        }
        return result
