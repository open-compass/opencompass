import os
import tempfile
from typing import Any, Dict, Iterable

import requests
from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils.datasets_info import DATASETS_MAPPING

from .base import BaseDataset


# Both MedFailBench artefacts below are pinned to one upstream commit so that a
# run is reproducible and the two files can be traced to a single revision.
MEDFAILBENCH_DATA_SHA = '7c6a9939bf6db67e7abd95a383e5aec229c5770d'

# Canonical MedFailBench v0.2 prompt set (222 rows, English + Turkish). This is
# the file that backs the official leaderboard and the LM Evaluation Harness
# task in the upstream project.
MEDFAILBENCH_LEADERBOARD_DATA_URL = (
    'https://raw.githubusercontent.com/goktugozkanmd/'
    'medical-ai-failure-atlas/'
    f'{MEDFAILBENCH_DATA_SHA}/'
    'leaderboard/medfailbench_prompts_v0_2.jsonl')

# Turkish-only safety-layer subset (44 rows). This is the preprint evidence set
# described in the upstream README and is NOT the leaderboard data file.
MEDFAILBENCH_TR_PREPRINT_DATA_URL = (
    'https://raw.githubusercontent.com/goktugozkanmd/'
    'medical-ai-failure-atlas/'
    f'{MEDFAILBENCH_DATA_SHA}/'
    'adapters/opencompass/medfailbench_safety_layer_docs_v0_1.jsonl')


def _is_url(path: str) -> bool:
    return path.startswith(('http://', 'https://'))


def _resolve_data_file(path: str, fallback_url: str) -> str:
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

    return fallback_url


def _medfailbench_cache_dir() -> str:
    cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
    if not cache_dir:
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache',
                                 'opencompass')
    return os.path.join(cache_dir, 'MedFailBench')


def _download_to_cache(url: str) -> str:
    """Download ``url`` once into the MedFailBench data cache."""
    cache_dir = _medfailbench_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, os.path.basename(url))
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target

    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix='.part')
    os.close(fd)
    try:
        headers = {'User-Agent': 'opencompass-medfailbench-adapter'}
        with requests.get(url, stream=True, timeout=60,
                          headers=headers) as response:
            response.raise_for_status()
            with open(tmp_path, 'wb') as fh:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
        os.replace(tmp_path, target)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return target


def _load_rows(path: str, fallback_url: str) -> Iterable[Dict[str, Any]]:
    data_file = _resolve_data_file(path, fallback_url)
    if _is_url(data_file):
        # The `json` dataset builder resolves data_files through fsspec
        # globbing, which does not work for plain raw.githubusercontent.com
        # URLs. Download once and hand the builder a local file instead.
        data_file = _download_to_cache(data_file)
    return load_dataset('json', data_files=data_file, split='train')


def _validate_leaderboard_rows(rows: Iterable[Dict[str, Any]]):
    """Validate the canonical v0.2 leaderboard prompt set."""
    normalized_rows = []
    required_fields = {
        'id',
        'prompt',
        'language',
        'domain',
        'safety_focus',
        'risk_axis',
        'safe_answer_hint',
        'category',
    }
    supported_languages = {'en', 'tr'}

    for row in rows:
        missing = required_fields.difference(row)
        if missing:
            raise ValueError(
                f'MedFailBench row {row.get("id", "<unknown>")} is missing '
                f'required fields: {sorted(missing)}')
        if row['language'] not in supported_languages:
            raise ValueError(
                f'MedFailBench row {row["id"]} has unsupported language '
                f'{row["language"]!r}; expected en or tr')

        normalized_rows.append(row)

    return normalized_rows


def _validate_tr_preprint_rows(rows: Iterable[Dict[str, Any]]):
    """Validate the Turkish-only v0.1 preprint evidence subset."""
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
                f'MedFailBench-TR-Preprint row {row.get("id", "<unknown>")} '
                f'is missing required fields: {sorted(missing)}')

        metadata = row.get('metadata') or {}
        if not metadata.get('synthetic_only'):
            raise ValueError(
                f'MedFailBench-TR-Preprint row {row["id"]} is not marked '
                f'synthetic_only')
        if metadata.get('contains_patient_data') is not False:
            raise ValueError(
                f'MedFailBench-TR-Preprint row {row["id"]} is not marked '
                f'patient-data-free')
        if row['language'] != 'tr':
            raise ValueError(
                f'MedFailBench-TR-Preprint row {row["id"]} has unsupported '
                f'language {row["language"]!r}')

        normalized_rows.append(row)

    return normalized_rows


@LOAD_DATASET.register_module()
class MedFailBenchDataset(BaseDataset):
    """MedFailBench v0.2: the 222-row bilingual leaderboard prompt set.

    Fields: ``id``, ``prompt``, ``language``, ``domain``, ``safety_focus``,
    ``risk_axis``, ``safe_answer_hint``, ``category``.
    """

    @staticmethod
    def load(path: str = MEDFAILBENCH_LEADERBOARD_DATA_URL):
        rows = _load_rows(path, MEDFAILBENCH_LEADERBOARD_DATA_URL)
        return Dataset.from_list(_validate_leaderboard_rows(rows))


@LOAD_DATASET.register_module()
class MedFailBenchTRPreprintDataset(BaseDataset):
    """MedFailBench-TR-Preprint v0.1: the 44-row Turkish evidence subset.

    Fields: ``id``, ``question``, ``target``, ``clinical_domain``,
    ``risk_axis``, ``safety_gate``, ``severity_1_to_5``, ``metadata``.
    """

    @staticmethod
    def load(path: str = MEDFAILBENCH_TR_PREPRINT_DATA_URL):
        rows = _load_rows(path, MEDFAILBENCH_TR_PREPRINT_DATA_URL)
        return Dataset.from_list(_validate_tr_preprint_rows(rows))
