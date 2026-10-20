# flake8: noqa
"""ELBench: a multi-dimensional benchmark for education-facing LLMs.

This module ports https://github.com/ZeroLoss-Lab/ELBench to OpenCompass while
keeping ELBench's original data files and naming untouched. It covers:

* Safety & Trustworthiness (安全可信)
    - 安全回答                          (LLM-as-a-Judge)
* High-Level Educational Cultivation (高阶育人)
    - 高阶育人-omni                    (objective, multiple choice)
    - 高阶育人-edu                     (LLM-as-a-Judge, response quality)
* General Capability (通用)
    - mmlu_pro / ceval                 (objective, multiple choice)
    - math_500 / aime24 / aime25 / aime26  (objective, math)
    - ifeval                           (rule-based instruction following)

Basic Education (基本教育) is a multi-turn teaching task; its data lives under
``benchmark_root/基本教育`` but it requires a multi-turn runtime and is not
wired into OpenCompass's single-turn pipeline here.
"""
import json
import os
import os.path as osp
import re
from collections import defaultdict

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (DICT_POSTPROCESSORS, ICL_EVALUATORS,
                                  LOAD_DATASET)
from opencompass.utils.datasets import get_data_path
from opencompass.utils.logging import get_logger

from ..base import BaseDataset
from .utils import get_judgeanswer_and_reference

# ---------------------------------------------------------------------------
# Data resolution.
#
# ELBench ships its files under a data root that holds the four original
# Chinese top-level module directories: 安全可信 / 通用 / 高阶育人 / 基本教育.
# The same layout is published on HuggingFace / ModelScope
# (ZeroLoss-Lab/ELBench) and registered in ``datasets_info.py`` as
# ``opencompass/ELBench`` -> ``./data/elbench``.
#
# Resolution mirrors ChemBench and other OC datasets:
#   * ``DATASET_SOURCE=ModelScope`` -> download the ModelScope snapshot;
#   * ``DATASET_SOURCE=HF``          -> download the HuggingFace snapshot;
#   * otherwise (default)            -> local mode, data must already be at
#                                        ``$COMPASS_DATA_CACHE/data/elbench``.
# No ELBench-specific env var is needed.
# ---------------------------------------------------------------------------

_ELBENCH_DATA_ROOT = None


def _elbench_data_root():
    """Return the local path to ELBench's data root directory."""
    global _ELBENCH_DATA_ROOT
    if _ELBENCH_DATA_ROOT and osp.isdir(_ELBENCH_DATA_ROOT):
        return _ELBENCH_DATA_ROOT
    logger = get_logger()

    dataset_source = os.environ.get('DATASET_SOURCE')
    resolved = get_data_path('opencompass/ELBench')

    if dataset_source == 'ModelScope':
        from modelscope import dataset_snapshot_download
        logger.info(f'ELBench: downloading from ModelScope {resolved}')
        root = dataset_snapshot_download(resolved)
    elif dataset_source == 'HF':
        from huggingface_hub import snapshot_download
        logger.info(f'ELBench: downloading from HuggingFace {resolved}')
        root = snapshot_download(repo_id=resolved, repo_type='dataset')
    else:
        # Local mode: ``resolved`` is already a filesystem path.
        root = resolved

    if not osp.isdir(root):
        raise FileNotFoundError(
            f'ELBench data root not found: "{root}". '
            f'Set COMPASS_DATA_CACHE to the parent of ``data/elbench``, or '
            f'use DATASET_SOURCE=HF/ModelScope to download automatically.')
    _ELBENCH_DATA_ROOT = root
    return root


def _elbench_path(subdir, name, ext):
    """Resolve an original ELBench data file by (subdir, name, extension)."""
    return osp.join(_elbench_data_root(), subdir, f'{name}.{ext}')


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Human-readable names for the ELBench Safety subsets.
ELBENCH_SAFETY_TASKS = {
    '安全回答': '应回答',
}


def _read_jsonl(filename):
    rows = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _norm_letters(text, options='ABCDEFGHIJ'):
    seen = []
    for ch in re.findall(r'[A-J]', str(text).upper()):
        if ch in options and ch not in seen:
            seen.append(ch)
    return ''.join(sorted(seen))


def _extract_choices(text, options='ABCDEFGHIJ'):
    """Extract one or more answer letters from a model response.

    Uses an explicit ``ANSWER:`` / ``答案：`` marker when present, otherwise
    falls back to the final line only if it consists solely of option letters
    and separators (so letters are never harvested from prose).
    """
    if not text:
        return ''
    # Explicit "ANSWER: X" / "答案：X" marker: take the tail after the marker and
    # keep only isolated option letters (never letters embedded in a word).
    marker = re.search(r'(?:ANSWER|答案|最终答案)\s*[:：是为]?\s*(.+)', text,
                       re.IGNORECASE)
    if marker:
        isolated = re.findall(r'(?<![A-Za-z])([A-Ja-j])(?![A-Za-z])',
                              marker.group(1))
        letters = _norm_letters(''.join(isolated), options)
        if letters:
            return letters
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].upper()
        # Accept only if the final line is just letters + separators/brackets.
        if _norm_letters(last, options) and not re.sub(
                r'[\sA-J,，、.。()\[\]:：和与]', '', last):
            return _norm_letters(last, options)
    return ''


def _extract_bracket_int(text, lo, hi):
    """Extract the integer inside the last ``[[n]]`` within ``[lo, hi]``."""
    if text is None:
        return None
    matches = re.findall(r'\[\[\s*(\d+)\s*\]\]', text)
    for raw in reversed(matches):
        value = int(raw)
        if lo <= value <= hi:
            return value
    return None


# ---------------------------------------------------------------------------
# Safety & Trustworthiness — subjective subset (安全回答)
# ---------------------------------------------------------------------------


@LOAD_DATASET.register_module()
class ELBenchSafetyDataset(BaseDataset):
    """Loader for the safety JSONL subset (LLM-as-a-Judge)."""

    def load(self, path: str, name: str, *args, **kwargs):
        raw_data = []
        for problem in _read_jsonl(_elbench_path(path, name, 'jsonl')):
            question = problem['question']
            raw_data.append({
                'question': question,
                'judge': {
                    'task': name,
                    'label': problem.get('label')
                    or problem.get('category', ''),
                    'subcategory': problem.get('subcategory', ''),
                    'question': question,
                },
            })
        return Dataset.from_list(raw_data)


def post_process_elbench_binary(judgement: dict):
    """Extract a binary 0/1 score from the judge output."""
    text = judgement['prediction']
    if text is None:
        return None
    value = _extract_bracket_int(text, 0, 1)
    if value is None:
        match = re.search(r'(?:score|评分|得分|分数)["\']?\s*[:：]\s*([01])', text,
                          re.IGNORECASE)
        if match is None:
            return None
        value = int(match.group(1))
    return {'score': value}


@DICT_POSTPROCESSORS.register_module('elbench_safety')
def elbench_safety_postprocess(output: dict, output_path: str) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        output, output_path, post_process_elbench_binary)
    if len(judged_answers) == 0:
        return {'error': 'No valid judgements were parsed.'}
    scores = defaultdict(list)
    for ans, ref in zip(judged_answers, references):
        task = ref.get('task', 'overall')
        scores['overall'].append(ans['score'])
        scores[ELBENCH_SAFETY_TASKS.get(task, task)].append(ans['score'])
    results = {t: 100 * sum(s) / len(s) for t, s in scores.items()}
    results['details'] = output
    return results


# ---------------------------------------------------------------------------
# High-Level Educational Cultivation
# ---------------------------------------------------------------------------


@LOAD_DATASET.register_module()
class ELBenchHighLevelOmniDataset(BaseDataset):
    """Loader for 高阶育人-omni (objective, single/multi choice)."""

    def load(self, path: str, name: str, *args, **kwargs):
        raw_data = []
        for problem in _read_jsonl(_elbench_path(path, name, 'jsonl')):
            answer = problem.get('answer', '')
            gold = ''.join(
                sorted(set(re.findall(r'[A-J]',
                                      str(answer).upper()))))
            raw_data.append({
                'question': problem['question'],
                'answer': gold,
            })
        return Dataset.from_list(raw_data)


@LOAD_DATASET.register_module()
class ELBenchHighLevelEduDataset(BaseDataset):
    """Loader for 高阶育人-edu (LLM-as-a-Judge, response quality)."""

    def load(self, path: str, name: str, *args, **kwargs):
        raw_data = []
        for problem in _read_jsonl(_elbench_path(path, name, 'jsonl')):
            question = problem.get('question') or problem.get('Question', '')
            reference = {
                k: problem[k]
                for k in ('Score', 'Scoring Details', 'Personalized Feedback')
                if k in problem
            }
            raw_data.append({
                'question': question,
                'judge': {
                    'task': 'highlevel_edu',
                    'subject': problem.get('Subject', ''),
                    'reference': json.dumps(reference, ensure_ascii=False),
                    'question': question,
                },
            })
        return Dataset.from_list(raw_data)


def post_process_elbench_scale(judgement: dict):
    """Extract a 1-10 quality score from the judge output."""
    text = judgement['prediction']
    value = _extract_bracket_int(text, 1, 10)
    if value is None:
        return None
    return {'score': value}


@DICT_POSTPROCESSORS.register_module('elbench_highlevel_edu')
def elbench_highlevel_edu_postprocess(output: dict, output_path: str) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        output, output_path, post_process_elbench_scale)
    if len(judged_answers) == 0:
        return {'error': 'No valid judgements were parsed.'}
    scores = [a['score'] for a in judged_answers]
    # Report on a 0-100 scale (1-10 -> 10-100).
    return {
        'score': 10 * sum(scores) / len(scores),
        'details': output,
    }


# ---------------------------------------------------------------------------
# General Capability — objective subsets (inspect-style JSONL)
# ---------------------------------------------------------------------------


@LOAD_DATASET.register_module()
class ELBenchGeneralDataset(BaseDataset):
    """Loader for the 通用 subsets (mmlu_pro / ceval / math / aime / ifeval)."""

    def load(self, path: str, name: str, *args, **kwargs):
        raw_data = []
        for problem in _read_jsonl(_elbench_path(path, name, 'jsonl')):
            messages = problem.get('input') or []
            if isinstance(messages, list) and messages:
                question = messages[0].get('content', '')
            else:
                question = str(messages)
            target = problem.get('target', '')
            raw_data.append({
                'question': question,
                'target': str(target),
            })
        return Dataset.from_list(raw_data)


@ICL_EVALUATORS.register_module()
class ELBenchChoiceEvaluator(BaseEvaluator):
    """Exact-set match for single/multiple choice answers."""

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        details = []
        correct = 0
        for pred, ref in zip(predictions, references):
            pred_letters = _extract_choices(pred)
            gold = ''.join(sorted(set(re.findall(r'[A-J]', str(ref).upper()))))
            ok = pred_letters == gold and gold != ''
            correct += int(ok)
            details.append({
                'pred': pred_letters,
                'answer': gold,
                'correct': ok
            })
        return {
            'accuracy': 100 * correct / len(predictions),
            'details': details,
        }


@ICL_EVALUATORS.register_module()
class ELBenchMathEvaluator(BaseEvaluator):
    """Boxed-answer match for math subsets (math_500 / aime)."""

    def score(self, predictions, references):
        from opencompass.datasets.math import (extract_boxed_answer,
                                               normalize_final_answer)
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        def norm(text):
            boxed = extract_boxed_answer(str(text))
            if boxed is None:
                boxed = str(text)
            return normalize_final_answer(boxed)

        details = []
        correct = 0
        for pred, ref in zip(predictions, references):
            p, r = norm(pred), norm(ref)
            ok = p == r and r != ''
            correct += int(ok)
            details.append({'pred': p, 'answer': r, 'correct': ok})
        return {
            'accuracy': 100 * correct / len(predictions),
            'details': details,
        }
