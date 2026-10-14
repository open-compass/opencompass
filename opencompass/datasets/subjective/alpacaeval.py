# flake8: noqa: E501
import json
import os.path as osp
from collections import defaultdict

from datasets import Dataset, DatasetDict

from opencompass.datasets.subjective.compass_arena_subjective_bench import \
    get_element_counts
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
                    },
                })
        dataset = Dataset.from_list(raw_data)
        return dataset


def post_process_alpacav2(completion: str):
    r"""Parse a completion that contains 'm' or 'M' and returns the rank of the
    model1.

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
def alpacaeval_postprocess(
    output: dict,
    output_path: str,
) -> dict:

    judged_answers, references = get_judgeanswer_and_reference(
        result=output,
        filename=output_path,
        post_process=post_process_alpacav2,
    )

    if len(judged_answers) == 0:
        scores = None

    win_model1, win_model2, categories = (
        defaultdict(float),
        defaultdict(float),
        defaultdict(float),
    )

    if 'base_models' in references[0]:
        base_models = references[0]['base_models']
    else:
        # TODO: Assuming the first model in the first record to be the base model
        # Might not necessarily be the case if infer_order=="random"
        base_models = [references[0]['answer1']]

    if isinstance(base_models, str):
        base_models = [base_models]

    for judged_answer, reference in zip(judged_answers, references):
        categories['total'] += 1
        categories[reference['capability']] += 1
        if judged_answer['rank'] == 1:
            if reference['answer1'] in base_models:
                win_model1[reference['capability']] += 1
                win_model1['total'] += 1
            else:
                win_model2[reference['capability']] += 1
                win_model2['total'] += 1
        else:
            if reference['answer1'] in base_models:
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


@DICT_POSTPROCESSORS.register_module('alpacaeval_bradleyterry')
def alpacaeval_bradleyterry_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        result=output,
        filename=output_path,
        post_process=post_process_alpacav2,
    )

    if 'prediction1' not in references[0]:
        raise ValueError(
            'prediction1 not in references. Set `keep_predictions=True` for LMEvaluator in dataset config and retry.'
        )

    if 'prediction2' not in references[0]:
        raise ValueError(
            'prediction2 not in references. Set `keep_predictions=True` for LMEvaluator in dataset config and retry.'
        )

    if 'base_models' in references[0]:
        base_models = references[0]['base_models']
    else:
        # TODO: Assuming the first model in the first record to be the base model
        # Might not necessarily be the case if infer_order=="random"
        base_models = [references[0]['answer1']]

    if isinstance(base_models, str):
        base_models = [base_models]

    results = {}
    matches = []
    for judged_answer, reference in zip(judged_answers, references):
        cur_dict = {}

        if judged_answer['rank'] == 1:
            if reference['answer1'] in base_models:
                cur_dict['winner'] = 'model_a'
            else:
                cur_dict['winner'] = 'model_b'
        elif judged_answer['rank'] == 2:
            if reference['answer1'] in base_models:
                cur_dict['winner'] = 'model_b'
            else:
                cur_dict['winner'] = 'model_a'
        else:
            cur_dict['winner'] = 'tie'

        cur_dict['capability'] = reference['capability']
        cur_dict['model_a'] = reference['answer1']
        cur_dict['model_b'] = reference['answer2']
        cur_dict['prediction1'] = reference['prediction1']
        cur_dict['prediction2'] = reference['prediction2']

        matches.append(cur_dict)

    ### ---------- Add Style Metadata ---------- ###
    matches = get_element_counts(
        data=matches,
        column='prediction1',
        suffix='_a',
    )
    matches = get_element_counts(
        data=matches,
        column='prediction2',
        suffix='_b',
    )

    results['matches'] = matches
    # results["details"] = output

    return results
