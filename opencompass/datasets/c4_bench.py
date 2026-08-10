"""C4 Bench dataset and evaluator.

Paper: https://arxiv.org/abs/2608.06501
Dataset: https://huggingface.co/datasets/sci-m-wang/C4-Eval

Citation::

    @misc{wang2026mllmsdecodecreativeleap,
          title={Can MLLMs Decode the Creative Leap? Introducing C4 for
                 Cross-Concept Understanding},
          author={Ming Wang and Yuqing Zhang and Tingna Xie and Xiangju Li and
                  Xiaocui Yang and Daling Wang and Shi Feng and Yifei Zhang},
          year={2026},
          eprint={2608.06501},
          archivePrefix={arXiv},
          primaryClass={cs.AI},
          url={https://arxiv.org/abs/2608.06501},
    }
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

DEFAULT_REPO_ID = 'sci-m-wang/C4-Eval'
DEFAULT_DATA_FILENAME = 'data/eval.jsonl'
PRIMARY_TASKS = ('H0', 'H1', 'H4', 'E0')
EXPLANATION_TASKS = ('E0', 'E1')
TASK_SPLITS = {
    'primary': PRIMARY_TASKS,
    'all': (*PRIMARY_TASKS, 'E1'),
    'H0': ('H0', ),
    'H1': ('H1', ),
    'H4': ('H4', ),
    'E0': ('E0', ),
    'E1': ('E1', ),
}

PUNCT_RE = re.compile(r'[\s\n\r\t，,。.!！?？:：;；、\'"“”‘’`·]+')
ANSWER_PATTERNS = (
    re.compile(
        r'["\']?answer["\']?\s*[:：]\s*["“”\']?([\u4e00-\u9fff]{4})',
        re.IGNORECASE,
    ),
    re.compile(r'(?:答案|成语)\s*(?:是|为|[:：])\s*["“”\']?([\u4e00-\u9fff]{4})'),
)


def normalize_c4_answer(text: str) -> str:
    return PUNCT_RE.sub('', str(text or '')).strip()


def _strip_code_fence(text: str) -> str:
    cleaned = str(text or '').strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()


def _recover_explicit_answer(text: str) -> str:
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(str(text or ''))
        if matches:
            return matches[-1]
    lines = [
        line.strip().strip('"“”') for line in str(text or '').splitlines()
        if line.strip()
    ]
    if lines and re.fullmatch(r'[\u4e00-\u9fff]{4}', lines[-1]):
        return lines[-1]
    return ''


def parse_c4_answer(task: str, output: str) -> Tuple[str, Optional[bool]]:
    if task in EXPLANATION_TASKS:
        try:
            parsed = json.loads(_strip_code_fence(output))
        except (json.JSONDecodeError, TypeError):
            return _recover_explicit_answer(output), False
        if not isinstance(parsed, dict):
            return '', False
        return str(parsed.get('answer', '')).strip(), True

    lines = [
        line.strip() for line in str(output or '').splitlines()
        if line.strip()
    ]
    if len(lines) == 1:
        answer = lines[0].strip('"“”')
    else:
        answer = _recover_explicit_answer(output)
    return answer, None


def _resolve_data_file(path: str, data_filename: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        local_path = os.path.join(path, data_filename)
        if os.path.isfile(local_path):
            return local_path

    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=path,
        filename=data_filename,
        repo_type='dataset',
    )


@LOAD_DATASET.register_module()
class C4BenchDataset(BaseDataset):
    """Load C4 task rows as structured multimodal ChatML messages."""

    @staticmethod
    def load(path: str = DEFAULT_REPO_ID,
             data_filename: str = DEFAULT_DATA_FILENAME,
             split: str = 'primary') -> DatasetDict:
        if split not in TASK_SPLITS:
            raise ValueError(f'Unsupported C4 Bench split: {split}')

        data_path = _resolve_data_file(path, data_filename)
        with open(data_path, encoding='utf-8') as file:
            rows = [json.loads(line) for line in file if line.strip()]

        selected = []
        for row in rows:
            if row['task'] not in TASK_SPLITS[split]:
                continue
            reference = {
                'task': row['task'],
                'answer': row['answer'],
                'answer_aliases': row.get('answer_aliases', []),
            }
            selected.append({
                'instance_id':
                row['instance_id'],
                'task':
                row['task'],
                'question':
                row['question'],
                'answer':
                row['answer'],
                'reference':
                reference,
                'chatml_question': [{
                    'role':
                    'user',
                    'content': [
                        {
                            'type': 'image',
                            'image_url': row['image'],
                        },
                        {
                            'type': 'text',
                            'text': row['question'],
                        },
                    ],
                }],
                'chatml_answer': [row['answer']],
            })

        dataset = Dataset.from_list(selected)
        return DatasetDict({'test': dataset, 'train': dataset})


@ICL_EVALUATORS.register_module()
class C4BenchEvaluator(BaseEvaluator):
    """Official exact-recovery and JSON-validity metrics for C4 Bench."""

    def score(self, predictions: List[str], references: List[Dict]) -> Dict:
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        records = []
        for prediction, reference in zip(predictions, references):
            task = str(reference['task'])
            parsed, json_valid = parse_c4_answer(task, prediction)
            aliases = reference.get('answer_aliases', []) or []
            accepted = {
                normalize_c4_answer(answer)
                for answer in [reference['answer'], *aliases]
                if normalize_c4_answer(answer)
            }
            records.append({
                'task': task,
                'exact': normalize_c4_answer(parsed) in accepted,
                'json_valid': json_valid,
            })

        metrics: Dict[str, float] = {}
        primary = [
            record for record in records if record['task'] in PRIMARY_TASKS
        ]
        if primary:
            metrics['Primary Score'] = 100 * sum(
                record['exact'] for record in primary) / len(primary)

        for task in TASK_SPLITS['all']:
            subset = [record for record in records if record['task'] == task]
            if not subset:
                continue
            if task != 'E1':
                metrics[f'{task} Exact Match'] = 100 * sum(
                    record['exact'] for record in subset) / len(subset)
            if task in EXPLANATION_TASKS:
                metrics[f'{task} JSON Valid'] = 100 * sum(
                    bool(record['json_valid'])
                    for record in subset) / len(subset)
        return metrics
