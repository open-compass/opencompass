# flake8: noqa
# yapf: disable
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmengine import ConfigDict
from tabulate import tabulate

from opencompass.utils import model_abbr_from_cfg

from .compass_arena import CompassArenaSummarizer
from .utils import get_judgeanswer_and_reference, get_outdir

COLUMNS = ['total', 'writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']

def model_abbr_from_cfg_used_in_summarizer(model):
    if model.get('summarizer_abbr', None):
        return model['summarizer_abbr']
    else:
        return model_abbr_from_cfg(model)

def post_process_mtbench_pair(judgement: str):
    """Input a string like below:

    xxx[[A]]xxx, and extract the judge
    """
    pattern = r'\[([A-C]+)\]'
    matched_result = re.findall(pattern, judgement)
    if matched_result:
        return matched_result[0]
    else:
        return None


def post_process_mtbench_single(judgement: str):
    """Input a string like below:

    xxx[[5]]xxx, and extract the score
    """
    pattern = r'Rating:\s*\[\[([\d.]+)\]\]'
    matched_result = re.findall(pattern, judgement)
    if matched_result:
        score = float(matched_result[0])
    else:
        return None
    return {'score': score}


def get_capability_results(
    judged_answers,
    references,
    fout,
    fout_flag,
    model_abbr,
):
    columns = COLUMNS
    capability_ratings = defaultdict(int)
    capability_counts = defaultdict(int)
    capability_avg_ratings = defaultdict(float)
    if len(judged_answers) == 0:
        for column in columns:
            capability_avg_ratings[column] = ''
    else:
        for ans, ref in zip(judged_answers, references):
            capability_ratings['total'] += ans['score']
            capability_counts['total'] += 1
            capability_ratings[ref['capability']] += ans['score']
            capability_counts[ref['capability']] += 1

        for capability, total_score in capability_ratings.items():
            s = total_score / capability_counts[capability]
            s = round(s, 2)
            capability_avg_ratings[capability] = s

    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            writer.writerow(['model'] + columns)
        writer.writerow([model_abbr] + [capability_avg_ratings[column] for column in columns])


class MTBenchSummarizer(CompassArenaSummarizer):
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
        elif self.judge_type == 'pair':
            self.base_models = self.cfg['eval']['partitioner']['base_models']
            self.compare_models = self.cfg['eval']['partitioner']['compare_models']
        self.judge_models = self.cfg.get('judge_models', None)
        self.judge_map = {
            'single': post_process_mtbench_single,
            'pair': post_process_mtbench_pair
        }
        self.judge_function = self.judge_map[self.judge_type]

    def summarize(self, time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        if self.judge_type == 'pair':
            return super().summarize()

        # self.judge_type == 'single'
        dataset_cfgs = self.cfg['datasets']
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        all_scores = {}
        for judge_model in self.judge_models:
            fout_flag = 0
            score_by_judgemodel = {}
            judge_abbr = model_abbr_from_cfg(judge_model)
            for eval_model_cfg in self.eval_model_cfgs:
                eval_model_abbr = model_abbr_from_cfg(eval_model_cfg)
                show_model_abbr = model_abbr_from_cfg_used_in_summarizer(eval_model_cfg)
                subdir_path = os.path.join(results_folder, eval_model_abbr + '_judged-by--' + judge_abbr)
                if os.path.isdir(subdir_path):
                    fout = osp.join(output_dir, 'MTBench-judged-by--' + judge_abbr + '-capability.csv')
                    overall_judged_answers, overall_references = [], []
                    for dataset in dataset_cfgs:
                        judged_answers, references = get_judgeanswer_and_reference(dataset, subdir_path, self.judge_function)
                        overall_judged_answers += judged_answers
                        overall_references += references
                    get_capability_results(overall_judged_answers, overall_references, fout, fout_flag, show_model_abbr)
                    fout_flag += 1
                else:
                    print(subdir_path + ' is not exist! please check!')
            with open(fout, 'r') as f:
                csv_reader = csv.reader(f)
                header = next(csv_reader)
                table = [line for line in csv_reader]

            for model_score in table:
                score_by_judgemodel[model_score[0]] = {}
                for idx, column in enumerate(COLUMNS):
                    score_by_judgemodel[model_score[0]][column] = model_score[idx+1]
            all_scores[judge_abbr] = score_by_judgemodel
        return {'MTbench': all_scores}
