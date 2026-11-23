import json
from typing import Iterable, Optional

from datasets import Dataset, load_dataset

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class General365Dataset(BaseDataset):
    """Load and optionally filter the official General365 public split."""

    @staticmethod
    def load(path: str = 'meituan-longcat/General365_Public',
             split: str = 'test',
             answer_types: Optional[Iterable[str]] = None,
             **kwargs) -> Dataset:
        dataset = load_dataset(path=path, split=split, **kwargs)
        if answer_types is not None:
            answer_types = set(answer_types)
            dataset = dataset.filter(
                lambda item: item['answer_type'] in answer_types)
        return dataset


def parse_general365_judgement(judgement: str):
    """Parse the JSON response required by the official GPT judge prompt."""
    if not isinstance(judgement, str):
        return None
    candidates = [judgement.strip()]
    if '```' in judgement:
        candidates.extend(part.strip() for part in judgement.split('```')
                          if part.strip())
    for candidate in candidates:
        if candidate.lower().startswith('json'):
            candidate = candidate[4:].strip()
        try:
            result = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        accuracy = result.get('accuracy') if isinstance(result, dict) else None
        if isinstance(accuracy, bool):
            return accuracy
    return None


@DICT_POSTPROCESSORS.register_module()
def general365_llmjudge_postprocess(output: dict,
                                    output_path: str,
                                    dataset=None) -> dict:
    """Convert official ``{"accuracy": bool}`` judgements to OC metrics."""
    correct = 0
    parse_errors = 0
    for value in output.values():
        judgement = parse_general365_judgement(value.get('prediction', ''))
        value['correct'] = judgement is True
        value['judge_parsed'] = judgement
        if judgement is True:
            correct += 1
        elif judgement is None:
            parse_errors += 1

    total = len(output)
    return {
        'accuracy': correct / total * 100 if total else 0.0,
        'correct_count': correct,
        'incorrect_count': total - correct - parse_errors,
        'parse_error_count': parse_errors,
        'total': total,
        'details': output,
    }
