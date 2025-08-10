# flake8: noqa: E501
import re
from collections import defaultdict
from typing import Optional

from opencompass.registry import DICT_POSTPROCESSORS

from .utils import get_judgeanswer_and_reference


def post_process(judgement: str):
    """Input a string like below:

    xxx[[5]]xxx, and extract the score
    """
    judgement = judgement['prediction']
    pattern = r'\[\[([\d.]+)\]\]'
    matched_result = re.findall(pattern, judgement)
    if matched_result:
        score = float(matched_result[0])
    else:
        return None
    return {'score': score}


def get_capability_results(judged_answers, references):
    capability_ratings = defaultdict(int)
    capability_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        capability_ratings['total'] += ans['score']
        capability_counts['total'] += 1
        capability_ratings[ref['capability']] += ans['score']
        capability_counts[ref['capability']] += 1

    capability_avg_ratings = defaultdict(float)

    for capability, total_score in capability_ratings.items():
        s = total_score / capability_counts[capability]
        s = round(s, 2)
        capability_avg_ratings[capability] = s

    return capability_avg_ratings


@DICT_POSTPROCESSORS.register_module('commenbench')
def commonbench_postprocess(
    output: dict,
    output_path: str,
    post_process: Optional[callable] = post_process,
) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        output, output_path, post_process)

    results = get_capability_results(judged_answers, references)
    results['details'] = output
    return results
