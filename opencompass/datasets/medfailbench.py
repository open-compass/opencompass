import os
from typing import Any, Dict, Iterable

from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils.datasets_info import DATASETS_MAPPING

from .base import BaseDataset


MEDFAILBENCH_DATA_URL = (
    'https://raw.githubusercontent.com/goktugozkanmd/'
    'medical-ai-failure-atlas/'
    '7c6a9939bf6db67e7abd95a383e5aec229c5770d/'
    'adapters/opencompass/medfailbench_safety_layer_docs_v0_1.jsonl')


def _is_url(path: str) -> bool:
    return path.startswith(('http://', 'https://'))


def _resolve_medfailbench_data_file(path: str) -> str:
    if _is_url(path) or os.path.isabs(path) or os.path.exists(path):
        return path

    mapping = DATASETS_MAPPING.get(path)
    if mapping:
        local_path = mapping.get('local')
        if local_path:
            cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
            candidates = [
                os.path.join(cache_dir, local_path),
                os.path.join(os.path.expanduser('~'), '.cache', 'opencompass',
                             local_path),
                local_path,
            ]
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    return candidate

    return MEDFAILBENCH_DATA_URL


def _validate_and_normalize(rows: Iterable[Dict[str, Any]]):
    normalized_rows = []
    required_fields = {
        'id',
        'language',
        'question',
        'target',
        'clinical_domain',
        'risk_axis',
        'safety_gate',
        'severity_1_to_5',
    }

    for row in rows:
        missing = required_fields.difference(row)
        if missing:
            raise ValueError(
                f'MedFailBench row {row.get("id", "<unknown>")} is missing '
                f'required fields: {sorted(missing)}')

        metadata = row.get('metadata') or {}
        if not metadata.get('synthetic_only'):
            raise ValueError(
                f'MedFailBench row {row["id"]} is not marked synthetic_only')
        if metadata.get('contains_patient_data') is not False:
            raise ValueError(
                f'MedFailBench row {row["id"]} is not marked patient-data-free'
            )
        if row['language'] != 'tr':
            raise ValueError(
                f'MedFailBench row {row["id"]} has unsupported language '
                f'{row["language"]!r}')

        normalized_rows.append(row)

    return normalized_rows


@LOAD_DATASET.register_module()
class MedFailBenchDataset(BaseDataset):
    """Turkish synthetic medical safety failure prompts."""

    @staticmethod
    def load(path: str = MEDFAILBENCH_DATA_URL):
        data_file = _resolve_medfailbench_data_file(path)
        dataset = load_dataset('json', data_files=data_file, split='train')
        return Dataset.from_list(_validate_and_normalize(dataset))
