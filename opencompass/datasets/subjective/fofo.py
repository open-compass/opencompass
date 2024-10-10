# flake8: noqa
import json
import os.path as osp
import re
from collections import defaultdict

from datasets import Dataset

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import get_judgeanswer_and_reference


@LOAD_DATASET.register_module()
class FofoDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.json')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                question = problem['instruction']
                lan = 'cn' if 'cn' in name else 'en'
                raw_data.append({
                    'question': question,
                    'judge': {
                        'lan': lan,
                        'id': problem['id'],
                        'domain': problem['domain'],
                        'sub_domain': problem['sub_domain'],
                        'format': problem['format'],
                        'format_type': problem['format_type'],
                        'question': question
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset


def post_process_fofo(judgement: dict):
    """Input a string like below:

    xxx[[5]]xxx, and extract the score
    """
    match = re.search(r"[\"']format_correctness[\"']:\s*([0-1]+)",
                      judgement['prediction'])
    if match:
        score = int(match.group(1))
    else:
        return None

    return {'score': score}


@DICT_POSTPROCESSORS.register_module('fofo')
def fofo_postprocess(output: dict, output_path: str) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        output, output_path, post_process_fofo)

    if len(judged_answers) == 0:
        scores = None

    scores = defaultdict(list)
    for ans, ref in zip(judged_answers, references):
        domain = ref['domain']
        format_name = ref['format']
        format_type = ref['format_type']
        score = ans['score']
        if score is not None:
            scores['overall'].append(score)
            scores[domain].append(score)
            if format_type == 'general':
                scores[format_name].append(score)
    single_model_scores = {
        task: sum(score) / len(score)
        for task, score in scores.items()
    }
    results = single_model_scores
    results['details'] = output
    return results
