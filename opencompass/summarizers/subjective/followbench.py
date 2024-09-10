# flake8: noqa: E501
import csv
import os
import os.path as osp
import re
import statistics
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmengine import ConfigDict

try:
    from prettytable import from_csv
except ImportError:
    from_csv = None

from opencompass.utils import model_abbr_from_cfg

from .subjective_post_process import post_process_autoj, post_process_judgelm
from .utils import get_judgeanswer_and_reference_update, get_outdir


def post_process_followbench(item):
    generation, level = item['prediction'], item['gold']['level']
    try:
        satisfy = generation.strip('```').strip().split('\n')[-1]

        if level == 1:
            if 'YES' in satisfy:
                return 1, 1
            elif 'NO' in satisfy:
                return 0, 0
            else:
                raise Exception('Invalid evaluation for level 1.')
        else:
            satisfy_list = re.search(r'\[.*\]', satisfy)
            if satisfy_list:
                satisfy_list = eval(satisfy_list.group())
                if len(satisfy_list) == level:
                    num_true = 0
                    for i in satisfy_list:
                        if i == 'YES' or i == 'True':
                            num_true += 1
                        elif i in [
                                'NO', 'False', 'PARTIAL', 'MAYBE', 'UNKNOWN',
                                'N/A'
                        ]:
                            num_true += 0
                        else:
                            raise Exception('Invalid element in the list.')
                    return int(num_true == level), num_true / level
                else:
                    raise Exception('Invalid number of elements in the list.')
            else:
                raise Exception('Invalid list that cannot be parsed.')

    except Exception as e:
        return -1, -1


def get_scores(judged_answers, references):
    results = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    n_group = len(judged_answers) // 5
    n_groups = [n_group] * 5

    for judged_answer, reference in zip(judged_answers, references):
        if judged_answer[0] == -1:
            n_groups[reference['level'] - 1] -= 1
        else:
            results[0][reference['level'] - 1] += judged_answer[0]
            results[1][reference['level'] - 1] += judged_answer[1]

    for i in range(len(results)):
        for j in range(len(results[i])):
            if n_groups[j] != 0:
                results[i][j] = results[i][j] / n_groups[j]
            else:
                results[i][j] = 0
    temp_dict = {
        'HSR_AVG': statistics.mean(results[0]),
        'SSR_AVG': statistics.mean(results[1])
    }
    for idx, s in enumerate(results[0]):
        temp_dict[f'HSR_L{idx+1}'] = s
    for idx, s in enumerate(results[1]):
        temp_dict[f'SSR_L{idx+1}'] = s

    return temp_dict


class FollowBenchSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict) -> None:
        self.tasks = []
        self.cfg = config
        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.eval_model_abbrs = [
            model_abbr_from_cfg(model) for model in self.eval_model_cfgs
        ]
        self.judge_models = self.cfg.get('judge_models', None)

        self.judge_function = post_process_followbench

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        all_scores = {}
        for judge_model in self.judge_models:
            score_by_judgemodel = {}
            judge_abbr = model_abbr_from_cfg(judge_model)
            dataset_cfgs = self.cfg['datasets']
            dataset = dataset_cfgs[0]  # Alignbench just have only one subfile
            output_dir, results_folder = get_outdir(self.cfg, time_str)

            fout = osp.join(output_dir,
                            'followbench-judged-by--' + judge_abbr + '.csv')

            for eval_model_abbr in self.eval_model_abbrs:
                subdir = eval_model_abbr + '_judged-by--' + judge_abbr
                subdir_path = os.path.join(results_folder, subdir)
                model = eval_model_abbr
                if os.path.isdir(subdir_path):
                    judged_answers, references = get_judgeanswer_and_reference_update(
                        dataset, subdir_path, self.judge_function)
                    if len(judged_answers) == 0:
                        score_by_judgemodel[model] = None
                        continue
                    scores = get_scores(judged_answers, references)
                    score_by_judgemodel[model] = scores
                else:
                    score_by_judgemodel[model] = None
                    print(subdir_path + ' is not exist! please check!')

            all_scores[judge_abbr] = score_by_judgemodel
        return {'followbench': all_scores}
