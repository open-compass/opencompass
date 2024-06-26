# flake8: noqa
# yapf: disable
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime
from itertools import product

import numpy as np
from mmengine import ConfigDict
from tabulate import tabulate

from opencompass.partitioners.sub_naive import remove_duplicate_pairs
from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg

from .compass_arena import (CompassArenaSummarizer, check_position_bias,
                            model_abbr_from_cfg_used_in_summarizer)
from .utils import get_judgeanswer_and_reference, get_outdir

task_group_new = {
    'Information seeking': 'Information/Advice seeking',
    'Creative Writing': 'Creative Tasks',
    'Coding & Debugging': 'Coding & Debugging',
    'Reasoning': 'Planning & Reasoning',
    'Editing': 'Creative Tasks',
    'Math': 'Math & Data Analysis',
    'Planning': 'Planning & Reasoning',
    'Brainstorming': 'Creative Tasks',
    'Role playing': 'Creative Tasks',
    'Advice seeking': 'Information/Advice seeking',
    'Data Analysis': 'Math & Data Analysis',
    'Others': 'Creative Tasks'}


def post_process_wildbench_pair(judgement: str):
    pattern = r'\"choice\": \"(.*?)\"'
    matched_result = re.findall(pattern, judgement)
    if matched_result:
        return matched_result[0]
    else:
        return None


def post_process_wildbench_single(judgement: str):
    pattern = r'\"score\": \"(.*?)\"'
    matched_result = re.findall(pattern, judgement)
    try:
        score = float(matched_result[0])
        return {'score': score}
    except (ValueError, IndexError) as e:
        return None

    # if matched_result:
    #     score = float(matched_result[0])
    # else:
    #     return None
    # return {'score': score}


def get_capability_results(
    judged_answers,
    references,
    fout,
    fout_flag,
    model_abbr,
):
    capability_ratings = defaultdict(float)
    capability_counts = defaultdict(float)

    for ans, ref in zip(judged_answers, references):
        # rescale
        capability_ratings['total'] += ans
        capability_counts['total'] += 1
        tags = [ref['primary_tag']] + ref['secondary_tag']
        for tag in tags:
            capability_ratings[task_group_new[tag]] += ans
            capability_counts[task_group_new[tag]] += 1

    capability_avg_ratings = defaultdict(float)

    for capability, total_score in capability_ratings.items():
        s = (total_score / capability_counts[capability] - 5) * 2 * 10
        s = round(s, 2)
        capability_avg_ratings[capability] = s
    columns = list(capability_avg_ratings.keys())
    columns.insert(0, columns.pop(columns.index('total')))

    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            writer.writerow(['model'] + columns)
        writer.writerow([model_abbr] + [capability_avg_ratings[column] for column in columns])


class WildBenchSingleSummarizer(CompassArenaSummarizer):
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict) -> None:
        self.judge_type = 'single'
        self.tasks = []
        self.cfg = config

        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_models'][0])
        self.judge_function = post_process_wildbench_single

    def summarize(self, time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """

        # self.judge_type == 'single'
        dataset_cfgs = self.cfg['datasets']
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        fout_flag = 0
        for eval_model_cfg in self.eval_model_cfgs:
            eval_model_abbr = model_abbr_from_cfg(eval_model_cfg)
            show_model_abbr = model_abbr_from_cfg_used_in_summarizer(eval_model_cfg)
            subdir_path = os.path.join(results_folder, eval_model_abbr + '_judged-by--' + self.judge_abbr)
            if os.path.isdir(subdir_path):
                fout = osp.join(output_dir, 'judged-by--' + self.judge_abbr + '-capability.csv')
                overall_judged_answers, overall_references = [], []
                for dataset in dataset_cfgs:
                    judged_answers, references = get_judgeanswer_and_reference(dataset, subdir_path, self.judge_function)
                    judged_answers = [item['score'] for item in judged_answers]
                    overall_judged_answers += judged_answers
                    overall_references += references

                get_capability_results(overall_judged_answers, overall_references, fout, fout_flag, show_model_abbr)
                fout_flag += 1
            else:
                print(subdir_path + ' is not exist! please check!')


class WildBenchPairSummarizer(CompassArenaSummarizer):
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, check_pos_bias=False) -> None:
        self.tasks = []
        self.cfg = config

        self.base_models = self.cfg['eval']['partitioner']['base_models']
        self.compare_models = self.cfg['eval']['partitioner']['compare_models']
        self.judge_models = self.cfg.get('judge_models', None)
        self.meta_judge_model = self.cfg.eval.partitioner.get('meta_judge_model', None)
        self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_models'][0])
        self.judge_function = post_process_wildbench_pair
        self.check_pos_bias = check_pos_bias

    def get_score(self, time_str):
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        model_combinations = list(product(self.base_models, self.compare_models))
        unique_combinations = remove_duplicate_pairs([combo for combo in model_combinations if combo[0] != combo[1]])

        if self.meta_judge_model is not None:
            self.judge_models.append(self.meta_judge_model)

        scores = {}
        for idx, judge_model_cfg in enumerate(self.judge_models):
            judge_model = model_abbr_from_cfg(judge_model_cfg)
            for dataset in self.cfg['datasets']:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                for model_pair in unique_combinations:
                    base_model = model_pair[0]['abbr']
                    compare_model = model_pair[1]['abbr']
                    if idx == len(self.judge_models):
                        subdir = base_model + '_' + compare_model + '_summarized-by--' + judge_model
                    else:
                        subdir = base_model + '_' + compare_model + '_judged-by--' + judge_model
                    subdir_path = os.path.join(results_folder, subdir)
                    if not os.path.isdir(subdir_path):
                        print(subdir_path + ' is not exist! please check!')
                        continue
                    judged_answers, references = get_judgeanswer_and_reference(dataset, subdir_path, self.judge_function)
                    if self.check_pos_bias:
                        bias_num = check_position_bias(judged_answers, references)
                    else:
                        bias_num = 0
                    win_base_model = defaultdict(float)
                    win_compare_model = defaultdict(float)
                    categories = defaultdict(float)
                    # base_model = references[0]['answer1']
                    # compare_model = references[0]['answer2']
                    score_mapping = {'A++': 1, 'A+': 0.5, 'A=B': 0, 'B+': -0.5, 'B++': -1}
                    for prediction, reference in zip(judged_answers, references):
                        if prediction not in score_mapping:
                            continue

                        categories[dataset_abbr] += 1
                        flag = 1 if reference['answer1'] == base_model else -1
                        score_1 = score_mapping[prediction]*flag
                        score_2 = -score_1

                        tags = [reference['primary_tag']] + reference['secondary_tag']
                        for tag in tags:
                            win_base_model[task_group_new[tag]] += score_1
                            win_compare_model[task_group_new[tag]] += score_2
                            categories[task_group_new[tag]] += 1

                        win_compare_model[dataset_abbr] += score_2
                        win_base_model[dataset_abbr] += score_1

                    for capability in categories:
                        win_base_model[capability] = win_base_model[capability] / categories[capability] * 100
                        win_base_model[capability] = round(win_base_model[capability], 2)
                        win_compare_model[capability] = win_compare_model[capability] / categories[capability] * 100
                        win_compare_model[capability] = round(win_compare_model[capability], 2)

                    win_base_model['position_bias'] = bias_num
                    win_compare_model['position_bias'] = bias_num

                    if judge_model not in scores:
                        scores[judge_model] = {}
                    if dataset_abbr not in scores[judge_model]:
                        scores[judge_model][dataset_abbr] = {}
                    scores[judge_model][dataset_abbr][base_model + '/' + compare_model] = win_compare_model

        return scores

    def summarize(
            self,
            time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S'),
    ):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        scores = self.get_score(time_str)
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        for idx, judge_model in enumerate(self.judge_models):
            judge_abbr = model_abbr_from_cfg(judge_model)
            for dataset in self.cfg['datasets']:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                summarizer_model_abbrs = [model_abbr_from_cfg_used_in_summarizer(i) for i in self.compare_models]
                one_column = list(scores[judge_abbr][dataset_abbr].values())[0]
                row_headers = [i for i in one_column.keys() if i not in [dataset_abbr, 'position_bias']]
                row_headers = [dataset_abbr, 'position_bias'] + row_headers

                table = []
                for row_header in row_headers:
                    row = [row_header]
                    headers = ['']
                    for model_cfg in self.compare_models:
                        model_abbr = model_abbr_from_cfg(model_cfg)
                        avg = 0
                        for base_model_cfg in self.base_models:
                            base_model_abbr = model_abbr_from_cfg(base_model_cfg)
                            base_compare = base_model_abbr + '/' + model_abbr
                            headers.append(base_compare)
                            s = scores[judge_abbr][dataset_abbr][base_compare].get(row_header, '')
                            if isinstance(s, float):
                                avg += s
                                s = f'{s:.2f}'
                            if isinstance(s, int):
                                s = str(s)
                            row.append(s)
                        avg = avg/len(self.base_models)
                        row.append(f'{avg:.2f}')
                        headers.append('Avg')
                    table.append(row)

                txt = tabulate(table, headers=headers)
                print(txt)

                if idx == len(self.judge_models):
                    output_filename = osp.join(output_dir, 'summarized-by--' + judge_abbr + '-' + dataset_abbr + '-report.csv')
                else:
                    output_filename = osp.join(output_dir, 'judged-by--' + judge_abbr + '-' + dataset_abbr + '-report.csv')

                with open(output_filename, 'w') as f:
                    f.write(','.join(headers) + '\n')
                    for line in table:
                        f.write(','.join(line) + '\n')
                print(output_filename)
