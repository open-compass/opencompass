# flake8: noqa: E501
import json
import os.path as osp
from collections import defaultdict

from datasets import Dataset, DatasetDict

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import get_judgeanswer_and_reference


@LOAD_DATASET.register_module()
class AlpacaEvalDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.json')
        dataset = DatasetDict()
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                question = problem['question']
                capability = problem['capability']
                others = problem['others']
                raw_data.append({
                    'question': question,
                    'capability': capability,
                    'others': others,
                    'judge': {
                        'capability': capability,
                        'question': question
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset


def post_process_alpacav2(completion: str):
    r"""Parse a completion that contains 'm' or 'M' and returns the rank of the model1.

    Examples
    --------
    >>> ranking_parser("m")
    1
    >>> ranking_parser("M")
    2
    >>> ranking_parser("s")
    None
    """
    completion = completion['prediction']
    try:
        if completion[0] == 'm':
            return {'rank': 1}
        elif completion[0] == 'M':
            return {'rank': 2}
        else:
            return None
    except Exception as e:
        return None


@DICT_POSTPROCESSORS.register_module('alpacaeval')
def alpacaeval_postprocess(output: dict, output_path: str) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        output, output_path, post_process_alpacav2)

    if len(judged_answers) == 0:
        scores = None

    win_model1, win_model2, categories = defaultdict(float), defaultdict(
        float), defaultdict(float)
    model1, model2 = references[0]['answer1'], references[0]['answer2']
    for prediction, reference in zip(judged_answers, references):
        categories['total'] += 1
        categories[reference['capability']] += 1
        if prediction['rank'] == 1:
            if reference['answer1'] == model1:
                win_model1[reference['capability']] += 1
                win_model1['total'] += 1
            else:
                win_model2[reference['capability']] += 1
                win_model2['total'] += 1
        else:
            if reference['answer1'] == model1:
                win_model2[reference['capability']] += 1
                win_model2['total'] += 1
            else:
                win_model1[reference['capability']] += 1
                win_model1['total'] += 1
    for capability in categories:
        if capability not in win_model1:
            win_model1[capability] = 0.0
        else:
            win_model1[capability] = round(
                (win_model1[capability] / categories[capability]) * 100, 2)
        if capability not in win_model2:
            win_model2[capability] = 0.0
        else:
            win_model2[capability] = round(
                (win_model2[capability] / categories[capability]) * 100, 2)

    results = win_model2
    results['details'] = output
    return results
