# Edited from the official SimpleQA config: https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py # noqa E501
import random
import re

import pandas
from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SimpleQADataset(BaseDataset):

    @staticmethod
    def load(path: str,
             num_examples: int | None = None,
             n_repeats: int = 1,
             **kwargs):
        path = get_data_path(path)
        dataset = DatasetDict()
        df = pandas.read_csv(path)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, \
                'n_repeats only supported when max_examples = None'
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        dataset['train'] = Dataset.from_list(examples)
        dataset['test'] = Dataset.from_list(examples)
        return dataset


def get_final_results(judged_answers, references, origial_responses):
    count = 0
    is_correct_count = 0
    is_incorrect_count = 0
    is_not_attempted_count = 0
    details = []
    for i, j, k in zip(judged_answers, references, origial_responses):
        match = re.search(r'(A|B|C)', i)
        grade_letter = match.group(
            0) if match else 'C'  # Default to "NOT_ATTEMPTED" if no match
        detail = {
            'pred': k,
            'ref': j,
            'origin_grade_response': i,
            'grade_letter': grade_letter,
            'correct': False
        }
        count += 1
        if grade_letter == 'A':
            is_correct_count += 1
            detail['correct'] = True
        elif grade_letter == 'B':
            is_incorrect_count += 1
        else:
            is_not_attempted_count += 1
        details.append(detail)

    is_correct = is_correct_count / count
    is_incorrect = is_incorrect_count / count
    # is_not_attempted = is_not_attempted_count / count
    is_given_attempted = is_correct + is_incorrect
    accuracy_given_attempted = is_correct / is_given_attempted \
        if is_given_attempted > 0 else 0
    f1 = 2 * accuracy_given_attempted * is_correct / (
        accuracy_given_attempted + is_correct) if (accuracy_given_attempted +
                                                   is_correct) > 0 else 0
    result = {
        'accuracy_given_attempted': accuracy_given_attempted,
        'f1': f1,
        'details': details
    }
    return result


def _single_simpleqa_postprocess(judgement: str):
    match = re.search(r'(A|B|C)', judgement)
    grade_letter = match.group(
        0) if match else 'C'  # Default to "NOT_ATTEMPTED" if no match
    return grade_letter


def simpleqa_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    judged_answers = []
    origial_responses = []
    references = []
    for k, v in output.items():
        origial_responses.append(v['prediction'])
        processed_judge = _single_simpleqa_postprocess(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            references.append(v['gold'])
    results = get_final_results(judged_answers, references, origial_responses)
    results['details'] = output
    return results
