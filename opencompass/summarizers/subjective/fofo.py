# flake8: noqa: E501
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmengine import ConfigDict
from tabulate import tabulate

try:
    from prettytable import from_csv
except ImportError:
    from_csv = None

from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg

from .compass_arena import CompassArenaSummarizer
from .utils import get_judgeanswer_and_reference, get_outdir

# from .utils.writer import Writer


def post_process_fofo(judgement: str):
    """Input a string like below:

    xxx[[5]]xxx, and extract the score
    """
    match = re.search(r"[\"']format_correctness[\"']:\s*([0-1]+)", judgement)
    if match:
        score = int(match.group(1))
    else:
        return None

    return {'score': score, 'judgement': judgement}


class FofoSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='single') -> None:

        self.tasks = []
        self.cfg = config

        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.eval_model_abbrs = [
            model_abbr_from_cfg(model) for model in self.eval_model_cfgs
        ]

        self.judge_models = self.cfg.get('judge_models', None)

        self.judge_function = post_process_fofo

    def get_score(self, time_str):
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        total_scores = {}
        for idx, judge_model_cfg in enumerate(self.judge_models):
            judge_model = model_abbr_from_cfg(judge_model_cfg)
            for dataset in self.cfg['datasets']:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                for eval_model_abbr in self.eval_model_abbrs:
                    subdir = eval_model_abbr + '_judged-by--' + judge_model
                    subdir_path = os.path.join(results_folder, subdir)
                    if os.path.isdir(subdir_path):
                        judged_answers, references = get_judgeanswer_and_reference(
                            dataset, subdir_path, self.judge_function)
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
                        if len(judged_answers) == 0:
                            single_model_scores = {}
                        else:
                            single_model_scores = {
                                task: sum(score) / len(score)
                                for task, score in scores.items()
                            }
                        if judge_model not in total_scores:
                            total_scores[judge_model] = {}
                        if dataset_abbr not in total_scores[judge_model]:
                            total_scores[judge_model][dataset_abbr] = {}
                        total_scores[judge_model][dataset_abbr][
                            eval_model_abbr] = single_model_scores
                    else:
                        print(subdir_path + ' is not exist! please check!')
        return total_scores

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        all_scores = {}
        scores = self.get_score(time_str)
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        for idx, judge_model in enumerate(self.judge_models):
            judge_abbr = model_abbr_from_cfg(judge_model)
            score_by_judgemodel = {}
            score_saver = {}
            for dataset in self.cfg['datasets']:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                summarizer_model_abbrs = self.eval_model_abbrs
                one_column = list(scores[judge_abbr][dataset_abbr].values())[0]
                format_types = ['Json', 'CSV', 'XML', 'YAML', 'Markdown']
                row_headers = [
                    i for i in one_column.keys()
                    if i not in [dataset_abbr] + format_types + ['overall']
                ]
                row_headers = ['overall'] + format_types + row_headers
                headers = [dataset_abbr] + summarizer_model_abbrs
                table = []
                for row_header in row_headers:
                    row = [row_header]
                    for model_abbr in summarizer_model_abbrs:
                        s = scores[judge_abbr][dataset_abbr][model_abbr].get(
                            row_header, '')
                        if isinstance(s, float):
                            s = f'{s:.2f}'
                        if isinstance(s, int):
                            s = str(s)
                        row.append(s)
                    table.append(row)
                txt = tabulate(table, headers=headers)
                score_saver[dataset_abbr] = [s for s in table[0][1:]]
                if idx == len(self.judge_models):
                    output_filename = osp.join(
                        output_dir, dataset_abbr + '-summarized-by--' +
                        judge_abbr + '-' + '-report.csv')
                else:
                    output_filename = osp.join(
                        output_dir, dataset_abbr + '-judged-by--' +
                        judge_abbr + '-' + '-report.csv')

                with open(output_filename, 'w') as f:
                    f.write(','.join(headers) + '\n')
                    for line in table:
                        f.write(','.join(line) + '\n')
            for idx, model in enumerate(summarizer_model_abbrs):
                score_by_judgemodel[model] = {}
                for subset_name, subset_scores in score_saver.items():
                    score_by_judgemodel[model][subset_name] = subset_scores[
                        idx]
            all_scores[judge_abbr] = score_by_judgemodel
        return {'Fofo': all_scores}
