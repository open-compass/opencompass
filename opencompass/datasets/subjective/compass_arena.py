# flake8: noqa: E501
import re
from collections import defaultdict

from datasets import Dataset

from opencompass.datasets.subjective.compass_arena_subjective_bench import \
    get_element_counts
from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET

from .subjective_cmp import SubjectiveCmpDataset
from .utils import get_judgeanswer_and_reference


@LOAD_DATASET.register_module()
class CompassArenaDataset(SubjectiveCmpDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        dataset = list(super().load(path, name))
        creation_dataset = []
        for data in dataset:
            if 'reference' in data['others']:
                if data['others']['reference'] is not None:
                    data['ref'] = data['others']['reference']
                else:
                    data['ref'] = '满足用户需求，言之有理即可'
            else:
                data['ref'] = '满足用户需求，言之有理即可'
            creation_dataset.append(data)
        dataset = Dataset.from_list(creation_dataset)
        return dataset


def check_position_bias(judged_answers, references, banned_choice=['C']):
    """Check position bias for judgellm's judgement.

    Args:
        judged_answers: The successfully extracted judgement.
        references: The references contains original question, which is used to located the same question for different position judgement.
    """
    position_bias_flag = 0
    position_bias_dict = {}
    for judge, ref in zip(judged_answers, references):
        question = ref['question']
        question_hash = hash(question)
        if question_hash not in position_bias_dict:
            position_bias_dict[question_hash] = {
                'question': question,
                'judge': judge
            }
        else:
            first_judge = position_bias_dict[question_hash]['judge']
            if (judge == first_judge and first_judge not in banned_choice
                    and judge not in banned_choice):
                # If second choice is same with first choice, there has position bias.
                position_bias_flag += 1
    return position_bias_flag


def post_process_compassarena(item):
    s = item['prediction']
    if result := re.findall('(?:选择：|Choice: )([ABC])', s):
        return result[0]
    else:
        return None


@DICT_POSTPROCESSORS.register_module('compassarena')
def compassarena_postprocess(
    output: dict,
    output_path: str,
    summary_type='single',
    check_pos_bias=True,
) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        output, output_path, post_process_compassarena)

    if check_pos_bias:
        bias_num = check_position_bias(judged_answers, references)
    else:
        bias_num = 0

    win_model1 = defaultdict(float)
    win_model2 = defaultdict(float)
    categories = defaultdict(float)
    model1 = references[0]['answer1']

    for prediction, reference in zip(judged_answers, references):

        categories[reference['capability']] += 1

        if prediction == 'A':
            if reference['answer1'] == model1:
                score_1, score_2 = 1, 0
            else:
                score_1, score_2 = 0, 1
        elif prediction == 'B':
            if reference['answer1'] == model1:
                score_1, score_2 = 0, 1
            else:
                score_1, score_2 = 1, 0
        elif prediction == 'C':
            if summary_type == 'half_add':
                score_1, score_2 = 0.5, 0.5
            else:
                score_1, score_2 = 0, 0

        win_model1[reference['capability']] += score_1
        win_model2[reference['capability']] += score_2
    for capability in categories:
        win_model1[
            capability] = win_model1[capability] / categories[capability] * 100
        win_model1[capability] = round(win_model1[capability], 2)
        win_model2[
            capability] = win_model2[capability] / categories[capability] * 100
        win_model2[capability] = round(win_model2[capability], 2)

    win_model1['position_bias'] = bias_num
    win_model2['position_bias'] = bias_num

    results = win_model2
    results['details'] = output
    return results


@DICT_POSTPROCESSORS.register_module('compassarena_bradleyterry')
def compassarena_bradleyterry_postprocess(
    output: dict,
    output_path: str,
    count_ties: bool = True,
) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        result=output,
        filename=output_path,
        post_process=post_process_compassarena,
    )

    if 'prediction1' not in references[0]:
        raise ValueError(
            'prediction1 not in references. Set `keep_predictions=True` for LMEvaluator in dataset config and retry.'
        )

    if 'prediction2' not in references[0]:
        raise ValueError(
            'prediction2 not in references. Set `keep_predictions=True` for LMEvaluator in dataset config and retry.'
        )

    results = {}
    matches = []
    for judged_answer, reference in zip(judged_answers, references):
        cur_dict = {}

        if judged_answer.strip() == 'A':
            cur_dict['winner'] = 'model_a'
        elif judged_answer.strip() == 'B':
            cur_dict['winner'] = 'model_b'
        elif judged_answer.strip() == 'C' and count_ties:
            cur_dict['winner'] = 'tie'
        else:
            continue

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
