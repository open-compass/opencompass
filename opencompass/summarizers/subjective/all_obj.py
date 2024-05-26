# flake8: noqa: E501
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmengine import ConfigDict
from prettytable import from_csv

from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg

from .utils import get_judgeanswer_and_reference, get_outdir


def post_process_allobj(judgement: str):
    """Input a string like below:

    xxx[[correct]]xxx, and extract the judge
    """
    pattern = r'(?i)\[(incorrect|correct|正确|错误|Yes|No)\]'
    matched_result = re.findall(pattern, judgement)
    if matched_result:
        content = matched_result[0].lower()
        if content in ['correct', '正确', 'yes']:
            return {'score': 1}
        elif content in ['incorrect', '错误', 'no']:
            return {'score': 0}
    else:
        return None


def get_capability_results(
    judged_answers,
    references,
    fout,
    fout_flag,
    model,
):
    capability_ratings = defaultdict(int)
    capability_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        capability_ratings['total'] += ans['score']
        capability_counts['total'] += 1

    capability_avg_ratings = defaultdict(float)

    for capability, total_score in capability_ratings.items():
        capability_avg_ratings[
            capability] = total_score / capability_counts[capability]
    columns = list(capability_avg_ratings.keys())
    columns.insert(0, columns.pop(columns.index('total')))
    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            writer.writerow(['model'] + columns)
        writer.writerow([model] +
                        [capability_avg_ratings[column] for column in columns])


class AllObjSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='single') -> None:
        self.judge_type = judge_type
        self.tasks = []
        self.cfg = config
        if self.judge_type == 'single':
            self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
            self.eval_model_abbrs = [
                model_abbr_from_cfg(model) for model in self.eval_model_cfgs
            ]
        elif self.judge_type == 'pair':
            self.base_models = self.cfg['eval']['partitioner']['base_models']
            self.compare_models = self.cfg['eval']['partitioner'][
                'compare_models']
        self.judge_abbr = model_abbr_from_cfg(
            self.cfg['eval']['partitioner']['judge_models'][0])
        self.judge_map = {'single': post_process_allobj}
        self.judge_function = self.judge_map[self.judge_type]

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        if self.judge_type == 'single':
            dataset_cfgs = self.cfg['datasets']
            judge_model = self.judge_abbr
            output_dir, results_folder = get_outdir(self.cfg, time_str)
            for dataset in dataset_cfgs:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                fout = osp.join(
                    output_dir,
                    'judged-by--' + judge_model + '-' + dataset_abbr + '.csv')
                fout_flag = 0
                for eval_model_abbr in self.eval_model_abbrs:
                    subdir = eval_model_abbr + '_judged-by--' + self.judge_abbr
                    subdir_path = os.path.join(results_folder, subdir)
                    if os.path.isdir(subdir_path):
                        model = eval_model_abbr
                        judged_answers, references = get_judgeanswer_and_reference(
                            dataset, subdir_path, self.judge_function)
                        get_capability_results(judged_answers, references,
                                               fout, fout_flag, model)
                        fout_flag += 1
                    else:
                        print(subdir_path + ' is not exist! please check!')
            with open(fout, 'r') as f:
                x = from_csv(f)
            print(x)
