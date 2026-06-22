import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CLBenchDataset(BaseDataset):
    """Load CLBench from HuggingFace and restore OpenCompass fields."""

    @staticmethod
    def load(path: str = 'tencent/CL-bench', split: str = 'train', **kwargs):
        load_kwargs = kwargs.copy()
        load_kwargs.pop('infer_cfg', None)
        load_kwargs.pop('eval_cfg', None)

        dataset = load_dataset(path=path, split=split, **load_kwargs)

        def enrich(example):
            metadata = example.get('metadata') or {}
            rubrics = example.get('rubrics') or []
            return {
                'rubrics_text':
                build_clbench_rubrics_text(rubrics),
                'task_id':
                example.get('task_id') or metadata.get('task_id'),
                'context_category': (example.get('context_category')
                                     or metadata.get('context_category')),
            }

        if isinstance(dataset, (Dataset, DatasetDict)):
            return dataset.map(enrich)
        return dataset


def build_clbench_rubrics_text(rubrics: Iterable[Any]) -> str:
    """Format CLBench rubrics as the official evaluator does."""
    if not rubrics:
        return 'No specific rubrics provided.'

    lines = []
    for i, rubric in enumerate(rubrics, 1):
        if isinstance(rubric, dict):
            criteria = str(rubric.get('rubric_criteria', '')).strip()
        else:
            criteria = str(rubric).strip()
        if criteria:
            lines.append(f'{i}. {criteria}')
    return '\n'.join(lines) if lines else 'No specific rubrics provided.'


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()


def _parse_overall_score(judgement: str) -> int:
    """Parse the binary Overall Score from a CLBench judge response."""
    text = _strip_json_fence(str(judgement))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict) and 'Overall Score' in parsed:
        try:
            return int(parsed['Overall Score'])
        except (TypeError, ValueError):
            return 0

    match = re.search(r'"?Overall Score"?\s*:\s*([01])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r'\b([01])\s*points?\b', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return 0


@DICT_POSTPROCESSORS.register_module()
def clbench_llmjudge_postprocess(
    output: Dict[str, Dict[str, Any]],
    output_path: str,
    dataset=None,
) -> Dict[str, Any]:
    """Postprocess CLBench LLM-judge results into OpenCompass metrics."""
    del output_path

    source_dataset = None
    if dataset is not None:
        source_dataset = dataset.reader.dataset['test']

    details: List[Dict[str, Any]] = []
    category_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        'correct': 0,
        'total': 0,
    })
    correct = 0

    for key in sorted(output, key=lambda x: int(x) if str(x).isdigit() else x):
        index = int(key) if str(key).isdigit() else len(details)
        judge_response = output[key].get('prediction', '')
        score = 1 if _parse_overall_score(judge_response) == 1 else 0

        sample = source_dataset[index] if source_dataset is not None else {}
        metadata = sample.get('metadata', {}) or {}
        category = (sample.get('context_category')
                    or metadata.get('context_category') or 'Unknown')
        task_id = sample.get('task_id') or metadata.get('task_id') or index
        model_output = sample.get('prediction', '')
        rubrics = sample.get('rubrics', [])

        correct += score
        category_stats[category]['correct'] += score
        category_stats[category]['total'] += 1

        details.append({
            'task_id': task_id,
            'context_category': category,
            'pred': model_output,
            'rubrics': rubrics,
            'origin_grade_response': judge_response,
            'score': score,
            'correct': bool(score),
        })

    total = len(details)
    accuracy = correct / total * 100 if total else 0
    category_accuracy = {
        category: stats['correct'] / stats['total'] * 100
        for category, stats in category_stats.items() if stats['total']
    }

    return {
        'accuracy': accuracy,
        'solving_rate': correct / total if total else 0,
        'correct_count': correct,
        'total_count': total,
        'category_accuracy': category_accuracy,
        'details': details,
    }
