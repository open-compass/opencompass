# flake8: noqa: E501
import ast
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime
from itertools import product

import mmengine
from mmengine import ConfigDict
from prettytable import from_csv

from opencompass.partitioners.sub_naive import remove_duplicate_pairs
from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg

from .utils import get_judgeanswer_and_reference, get_outdir


def post_process_compass_arena(s):
    if result := re.findall('(?:选择：|Choice: )([ABC])', s):
        return result[0]
    else:
        return None


class Compassarena_Summarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='general') -> None:
        self.tasks = []
        self.cfg = config
        self.base_models = self.cfg['eval']['partitioner']['base_models']
        self.compare_models = self.cfg['eval']['partitioner']['compare_models']
        self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_model'])
        self.judge_type = judge_type
        assert self.judge_type in ['general']
        self.judge_map = {
            'general': post_process_compass_arena,
        }
        self.judge_function = self.judge_map[self.judge_type]

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        dataset_cfgs = self.cfg['datasets']
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        model_combinations = list(
            product(self.base_models, self.compare_models))
        unique_combinations = remove_duplicate_pairs(
            [combo for combo in model_combinations if combo[0] != combo[1]])
        fout_list = []
        for model_pair in unique_combinations:
            model1, model2, judge_model = model_pair[0]['abbr'], model_pair[1][
                'abbr'], self.judge_abbr
            subdir = model1 + '_' + model2 + '_judged-by--' + self.judge_abbr
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                for dataset in dataset_cfgs:
                    dataset_abbr = dataset_abbr_from_cfg(dataset)
                    fout = osp.join(
                        output_dir, 'judged-by--' + judge_model + '-' +
                        dataset_abbr + '-report.csv')
                    fout_list.append(fout)
                    judged_answers, references, bias_num = get_judgeanswer_and_reference(
                        dataset,
                        subdir_path,
                        self.judge_function,
                        check_position_bias=True)
                    win_model1, win_model2, categories = defaultdict(
                        float), defaultdict(float), defaultdict(float)
                    model1, model2 = references[0]['answer1'], references[0][
                        'answer2']
                    for prediction, reference in zip(judged_answers,
                                                     references):
                        if dataset_abbr == 'zhihu_hot_0113':
                            reference['capability'] = 'QA'
                        categories['total'] += 1
                        categories[reference['capability']] += 1
                        if prediction == 'A':
                            if reference['answer1'] == model1:
                                win_model1[reference['capability']] += 1
                                win_model1['total'] += 1
                            else:
                                win_model2[reference['capability']] += 1
                                win_model2['total'] += 1
                        elif prediction == 'B':
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
                                (win_model1[capability] /
                                 categories[capability]) * 100, 2)
                        if capability not in win_model2:
                            win_model2[capability] = 0.0
                        else:
                            win_model2[capability] = round(
                                (win_model2[capability] /
                                 categories[capability]) * 100, 2)
                    win_model1['position_bias'] = bias_num
                    win_model2['position_bias'] = bias_num
                    scores = {
                        'win_' + model1: win_model1,
                        'win_' + model2: win_model2
                    }
                    rows = list(scores.keys())
                    columns = list(scores[rows[0]].keys())
                    columns.insert(0, columns.pop(columns.index('total')))
                    columns.insert(1,
                                   columns.pop(columns.index('position_bias')))
                    with open(fout, 'a+', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([model1 + '_vs_' + model2] + columns)
                        for row in rows:
                            writer.writerow(
                                [row] +
                                [scores[row][column] for column in columns])
            else:
                print(subdir_path + ' is not exist! please check!')
        for fout in fout_list:
            with open(fout, 'r') as f:
                x = from_csv(f)
            print(x)
