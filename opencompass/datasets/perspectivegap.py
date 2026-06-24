import json

from datasets import Dataset, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

_HF_DATASET = 'sun1245/PerspectiveGap'
_HF_DATA_FILE = 'evaluations.jsonl'


def _strip_think_tags(text: str) -> str:
    """Extract the answer after </think>, or return text as-is."""
    parts = text.rsplit('</think>', 1)
    if len(parts) == 2:
        return parts[1].strip()
    return text.strip()


def _resolve_data_file(path: str) -> str:
    path = path or _HF_DATASET
    if path.endswith('.jsonl') or path.startswith(('http://', 'https://')):
        return path
    return ('https://huggingface.co/datasets/'
            f'{path}/resolve/main/{_HF_DATA_FILE}')


def _reference_need_sets(row: dict) -> dict:
    raw = row['reference_need_sets']
    roles = row.get('roles') or raw.keys()
    return {
        role: raw[role]
        for role in roles if isinstance(raw.get(role), list)
    }


@LOAD_DATASET.register_module()
class PerspectiveGapDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        ds = load_dataset('json',
                          data_files={'test': _resolve_data_file(path)},
                          split='test')
        rows = []
        for row in ds:
            rows.append({
                'input':
                row[f'{name}_prompt'],
                'reference_data':
                json.dumps({
                    'reference_need_sets': _reference_need_sets(row),
                    'distractor_id': row['distractor_id'],
                    'fragments': row['fragments'],
                }),
            })
        return Dataset.from_list(rows)


@ICL_EVALUATORS.register_module()
class PerspectiveGapRoleAssignmentEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        from perspective_gap.scoring import score_role_assignment

        details = []
        cnt = 0
        for pred, ref_json in zip(predictions, references):
            pred = _strip_think_tags(pred)
            ref = json.loads(ref_json)
            result = score_role_assignment(
                pred,
                ref['reference_need_sets'],
                ref.get('distractor_id'),
            )
            is_correct = result['pass']
            if is_correct:
                cnt += 1
            details.append({
                'pred': pred[:200],
                'correct': is_correct,
                'metrics': result['metrics'],
            })

        score = cnt / len(predictions) * 100 if predictions else 0
        return {'score': score, 'details': details}


@ICL_EVALUATORS.register_module()
class PerspectiveGapPromptWritingEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        from perspective_gap.scoring import score_prompt_writing

        details = []
        cnt = 0
        for pred, ref_json in zip(predictions, references):
            pred = _strip_think_tags(pred)
            ref = json.loads(ref_json)
            result = score_prompt_writing(
                pred,
                ref['fragments'],
                ref['reference_need_sets'],
                ref.get('distractor_id'),
            )
            is_correct = result['pass']
            if is_correct:
                cnt += 1
            details.append({
                'pred': pred[:200],
                'correct': is_correct,
                'metrics': result['metrics'],
            })

        score = cnt / len(predictions) * 100 if predictions else 0
        return {'score': score, 'details': details}
