import re
from typing import Optional

from datasets import load_dataset

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
from opencompass.utils import extract_non_reasoning_content

from .base import BaseDataset

# `judge_points_max` of judges/arxiv_judge_gemini_38_flash in MathArena
JUDGE_POINTS_MAX = 3


@LOAD_DATASET.register_module()
class BrokenArXivDataset(BaseDataset):
    """BrokenArXiv monthly releases hosted on Hugging Face, e.g.
    ``MathArena/brokenarxiv-1026``."""

    @staticmethod
    def load(path: str, split: str = 'train', **kwargs):
        return load_dataset(path, split=split, **kwargs)


def parse_brokenarxiv_points(judgement: str) -> Optional[int]:
    """Port of the points parsing in MathArena ``SimpleJudge``: exactly one
    ``<points>`` tag holding an integer in [0, JUDGE_POINTS_MAX]."""
    if not isinstance(judgement, str):
        return None
    # MathArena parses the final judge response only, not its reasoning
    text = extract_non_reasoning_content(judgement)
    fields = re.findall(r'<points>(.*?)</points>',
                        text,
                        flags=re.IGNORECASE | re.DOTALL)
    if len(fields) != 1:
        return None
    match = re.fullmatch(r'\s*(?:Final\s+score:\s*)?([0-9]+)\s*', fields[0],
                         re.IGNORECASE)
    if match is None:
        return None
    points = int(match.group(1))
    if not 0 <= points <= JUDGE_POINTS_MAX:
        return None
    return points


@DICT_POSTPROCESSORS.register_module()
def brokenarxiv_llmjudge_postprocess(output: dict, output_path: str) -> dict:
    """Score = mean(points / 3), as in MathArena judge.py.

    MathArena retries an unparsable judgement up to three times and leaves it
    pending afterwards; here it counts as 0 points and is reported in
    ``parse_error_count``.
    """
    counts = [0] * (JUDGE_POINTS_MAX + 1)
    details = {}
    parse_error_count = 0
    for key, value in output.items():
        parsed = parse_brokenarxiv_points(value.get('prediction'))
        if parsed is None:
            parse_error_count += 1
        points = parsed or 0
        counts[points] += 1
        details[key] = {**value, 'points': points, 'judge_parsed': parsed}

    total = len(details)
    results = {
        'score':
        100 * sum(p * c for p, c in enumerate(counts)) /
        (JUDGE_POINTS_MAX * total) if total else 0,
    }
    for p, c in enumerate(counts):
        results[f'points_{p}_rate'] = 100 * c / total if total else 0
    results['parse_error_count'] = parse_error_count
    results['total'] = total
    results['details'] = details
    return results
