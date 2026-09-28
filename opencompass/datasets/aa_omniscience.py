import re
from collections import Counter
from typing import Optional

from datasets import load_dataset

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET

from .base import BaseDataset

DEFAULT_DATASET_PATH = 'ArtificialAnalysis/AA-Omniscience-Public'


@LOAD_DATASET.register_module()
class AAOmniscienceDataset(BaseDataset):
    """Public AA-Omniscience dataset hosted on Hugging Face."""

    @staticmethod
    def load(path: str = DEFAULT_DATASET_PATH, split: str = 'train', **kwargs):
        return load_dataset(path=path, split=split, **kwargs)


def _parse_aa_omniscience_judgement(judgement: str) -> Optional[str]:
    """Return the last explicit grade letter or word label in a judgement."""
    if not isinstance(judgement, str):
        return None

    letter_matches = list(
        re.finditer(r'(?<![A-Za-z])([ABCD])(?![A-Za-z])', judgement))
    if letter_matches:
        return letter_matches[-1].group(1)

    normalized = judgement.upper()
    word_pattern = re.compile(
        r'\b(?P<not_attempted>NOT[\s_-]+ATTEMPTED)\b|'
        r'\b(?P<partial>PARTIAL(?:LY)?[\s_-]+(?:ANSWER|CORRECT))\b|'
        r'\b(?P<incorrect>INCORRECT)\b|'
        r'\b(?P<correct>CORRECT)\b')
    word_matches = list(word_pattern.finditer(normalized))
    if not word_matches:
        return None

    return {
        'correct': 'A',
        'incorrect': 'B',
        'partial': 'C',
        'not_attempted': 'D',
    }[word_matches[-1].lastgroup]


@DICT_POSTPROCESSORS.register_module()
def aa_omniscience_llmjudge_postprocess(output: dict,
                                        output_path: str) -> dict:
    """Compute the official AA-Omniscience metrics from judge labels."""
    counts = Counter()
    details = {}
    parse_error_count = 0
    for key, value in output.items():
        judge_response = value.get('prediction', '')
        parsed_grade = _parse_aa_omniscience_judgement(judge_response)
        if parsed_grade is None:
            parse_error_count += 1
        grade_letter = parsed_grade or 'D'
        counts[grade_letter] += 1
        details[key] = {
            **value,
            'grade_letter': grade_letter,
            'judge_parsed': parsed_grade,
        }

    correct = counts['A']
    incorrect = counts['B']
    partial = counts['C']
    not_attempted = counts['D']
    total = correct + incorrect + partial + not_attempted
    non_correct = partial + incorrect + not_attempted

    return {
        'score':
        100 * (correct - incorrect) / total if total else 0,
        'accuracy':
        100 * correct / total if total else 0,
        'hallucination_rate':
        100 * incorrect / non_correct if non_correct else 0,
        'attempt_rate':
        100 * (correct + partial + incorrect) / total if total else 0,
        'correct_count':
        correct,
        'incorrect_count':
        incorrect,
        'partial_answer_count':
        partial,
        'not_attempted_count':
        not_attempted,
        'parse_error_count':
        parse_error_count,
        'total':
        total,
        'details':
        details,
    }
