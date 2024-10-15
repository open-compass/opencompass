# flake8: noqa: E501
import json

import numpy as np
from datasets import Dataset

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET

from ..base import BaseDataset

REGRESSION_DICT = {
    'travel': [
        14.816357060044538, 9.912640189521913, 6.854178078417421,
        16.548365732493735, 12.49440306294194, 19.925026350726633,
        19.449029525853824
    ],
    'Tech': [
        9.730382391494699, 15.439961810101806, 8.71267868266836,
        17.047912114497525, 8.188210881912578, 18.27285649160541,
        22.607997627719627
    ],
    'Sport': [
        10.470669731543392, 9.628138754444748, 5.8376755613192275,
        18.908737698203687, 11.170106247242, 22.555525595175727,
        21.42914641207122
    ],
    'Science': [
        10.215624094265426, 11.85130160404758, 13.199743482703303,
        15.771351181725294, 10.772433227719386, 18.259334358981764,
        19.93021205055725
    ],
    'music': [
        10.570558131445923, 10.250703197641212, 8.71555097518865,
        20.767746121844873, 15.130089494653312, 17.459999261696932,
        17.10535281752909
    ],
    'health': [
        14.409021815474166, 8.996654196952731, 9.058311451032425,
        20.374818020413127, 13.113089390107218, 13.622853268531996,
        20.425251857488348
    ],
    'write': [
        17.20178646947119, 11.647858398657238, 13.549614784591249,
        18.451414657788348, 8.415665936780018, 15.693785853424465,
        15.039873899287489
    ],
    'book': [
        10.987786546263385, 10.80583601777249, 11.110641533898052,
        21.917965372650762, 7.7575931269958955, 15.37978249492496,
        22.040394907494466
    ],
    'food': [
        10.88637513076461, 11.972608253231327, 13.762365658958538,
        18.449103701644535, 10.828866753473488, 15.403319360473219,
        18.69736114145427
    ],
    'movie': [
        13.987702750429126, 13.781107282170971, 10.70081300442185,
        14.950249677014197, 9.043151114164273, 14.990326778304123,
        22.54664939349545
    ],
    'long_dialogue': [
        12.655129633685263, 12.128629670452108, 15.417359033606798,
        8.805077038076321, 22.44683162734655, 19.3826287336546,
        9.164344263178364
    ],
    'blogs': [
        7.586054691386359, 19.535003668901773, 15.361732493611802,
        17.16924394200404, 16.86984484117092, 23.47812036292512
    ],
    'academic_article': [
        7.1513786821899865, 13.027210863148744, 17.517148962264663,
        14.222879878391684, 18.018026707391165, 30.06335490661375
    ],
    'report': [
        8.962021075489186, 17.645150656180856, 17.07695284575253,
        12.962529199816222, 18.77731391007885, 24.57603231268235
    ],
    'news': [
        5.746318852823619, 18.828108458188307, 18.348616241165825,
        16.546667215885762, 20.49878321641544, 20.03150601552105
    ],
    'question_generation': [
        15.630644520221793, 17.815836405315725, 5.260151108793491,
        5.260151108793491, 30.281435872156237, 25.751780984719254
    ],
    'character_creation': [
        13.387472615551518, 16.154170714995903, 5.3564749039425825,
        17.745872651899493, 27.8316766814783, 19.5243324321322
    ],
    'script_write': [
        16.020800876858075, 12.537284513149297, 7.583604904411543,
        10.962130120971509, 21.55253807214911, 31.343641512460472
    ],
    'report_write': [
        23.715406207770044, 11.322739017895511, 6.455129156251138,
        7.266817046605194, 20.795517896089837, 30.444390675388284
    ],
    'science_problem_solve': [
        14.532002727010074, 13.988091295206875, 5.78110629330191,
        13.906652976851941, 29.749526456076786, 22.042620251552407
    ],
    'academic_write': [
        18.274980968685334, 17.668799475735167, 5.373737221539396,
        15.33990358340595, 27.116855004727352, 16.225723745906805
    ],
    'guide_generation': [
        24.991645087603484, 12.995989180532902, 11.348066943331492,
        13.176536571757417, 19.238518079064633, 18.249244137710075
    ],
    'creative_write': [
        20.56735945510573, 13.865892755893375, 9.95947810767433,
        16.610586533096885, 21.307725530193018, 17.68895761803666
    ],
    'question_answering': [
        14.257396776453227, 12.59853746572811, 5.7410180060529985,
        15.959901439015228, 28.83810948056622, 22.60503683218423
    ],
    'curriculum_development': [
        20.68850512855878, 22.200461872620195, 5.8343282109082,
        5.8343282109082, 17.89639729448703, 27.545979282517592
    ],
    'continue_write': [
        18.669885223104068, 21.418933575454858, 13.841889274353397,
        6.502715042824038, 17.14288545529491, 22.423691428968727
    ],
    'idea_generation': [
        16.608491609592104, 24.45709647197801, 12.235414254617053,
        5.504078770891624, 18.79437075684626, 22.400548136074956
    ],
    'data_analysis': [
        18.29675276651988, 5.722157365550123, 5.740218388378298,
        20.92508664739828, 26.510684489335194, 22.80510034281823
    ],
    'rewrite': [
        20.801683025093183, 8.510828270810512, 11.130570080160155,
        13.722027611417639, 19.803701313664753, 26.03118969885375
    ],
    'explanation': [
        10.313604819556165, 18.631545950717513, 16.412914400566404,
        11.838586893660816, 19.111282531748692, 23.69206540375043
    ],
    'continuation': [
        21.427707308340644, 19.022158840412466, 16.220256947514333,
        20.57043807105919, 22.759438832673375
    ],
    'imitative_writing': [
        19.87078310837695, 19.793380163686955, 19.346176082395687,
        21.77086167116268, 19.218798974377737
    ],
    'style_transfer': [
        16.438886068023052, 18.226961726018953, 21.448441756584106,
        23.961762450033103, 19.923947999340776
    ],
    'story_writing': [
        23.5319284319259, 22.420937450120597, 10.539906363853124,
        17.047083302574496, 26.460144451525895
    ],
    'keyword_writing': [
        16.27370693012242, 27.30111800645728, 15.728682122621054,
        18.81389796206547, 21.882594978733778
    ],
    'screenplay_writing': [
        19.822086987393824, 20.973270981524056, 17.095645893112255,
        19.56592278203641, 22.543073355933444
    ],
    'argumentative_writing': [
        18.302865025230115, 24.50501277580138, 20.483643154138807,
        14.552018259438853, 22.15646078539085
    ],
    'roleplaying_writing': [
        18.23837535323756, 22.299189217994243, 12.860964861892231,
        19.918295740192793, 26.683174826683164
    ]
}


@LOAD_DATASET.register_module()
class HelloBenchDataset(BaseDataset):

    def load(self, path: str, category_name: str, *args, **kwargs):
        with open(f'{path}/{category_name}.jsonl', 'r', encoding='utf-8') as f:
            hellobench_dataset = [json.loads(line) for line in f.readlines()]
            for hellobench_dict in hellobench_dataset:
                hellobench_dict['judgement'] = {
                    'category': category_name,
                    'subcategory': hellobench_dict['category'],
                    'num_checklist': hellobench_dict['num_checklist']
                }
        dataset = Dataset.from_list(hellobench_dataset)
        return dataset


def post_process_hellobench(judgement):
    """Input a string like below:

    {'checklist_id': 0, 'reason': 'xxx', 'evaluation_score': 0.5}
    and extract each score
    """
    num_checklist = judgement['gold']['num_checklist']
    judgement = judgement['prediction']

    try:
        judgement = judgement.replace('```json',
                                      '').replace('```python',
                                                  '').replace('```', '')
        judgement = judgement.replace('\n', '').replace('\\', '')
        judgement_list = json.loads(judgement)
        return_list = []
        for judgement_dict in judgement_list:
            judgement_dict['checklist_id'] = int(
                judgement_dict['checklist_id'])
            judgement_dict['evaluation_score'] = float(
                judgement_dict['evaluation_score'])
            assert judgement_dict['evaluation_score'] <= 1.0 and judgement_dict[
                'evaluation_score'] >= 0.0
            return_list.append(judgement_dict['evaluation_score'])
        assert len(return_list) == num_checklist
        return return_list
    except:
        return None


def get_judgeanswer(result, filename, post_process):
    """Extract judgements (scores)

    Args:
        result (dict): result dict.
        filename (str): result path.
        post_process (function): The pre-defined extract function.
    """
    if len(result) == 0:
        print('*' * 100)
        print('There are no results for ' + filename)
        print('*' * 100)
    rescaled_score_dict = {}
    for k, v in result.items():
        processed_judge = post_process(v)

        if processed_judge is not None:
            subcategory = v['gold']['subcategory']
            weighted_dict = REGRESSION_DICT[subcategory]
            overall_score = np.dot(weighted_dict, processed_judge)
            rescaled_score = (overall_score - 75) * 4
            rescaled_score_dict[k] = rescaled_score

    if len(rescaled_score_dict) <= 0.95 * len(result):
        print('*' * 100)
        print(
            f'For your {filename} judge. Among {len(result)} judgements, successfully extracted {len(rescaled_score_dict)} judgements, please check!'
        )
        print('*' * 100)
    return rescaled_score_dict


@DICT_POSTPROCESSORS.register_module('hellobench')
def hellobench_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    rescaled_score_dict = get_judgeanswer(output, output_path,
                                          post_process_hellobench)

    results = {}
    results['overall_score'] = np.mean(list(rescaled_score_dict.values()))
    results['details'] = output

    for k, v in results['details'].items():
        if k in rescaled_score_dict:
            results['details'][k]['rescaled_score'] = rescaled_score_dict[k]
        else:
            results['details'][k]['rescaled_score'] = None

    return results
