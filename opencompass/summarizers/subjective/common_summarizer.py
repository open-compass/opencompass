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

from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg

from .compass_arena import CompassArenaSummarizer
from .utils import get_judgeanswer_and_reference, get_outdir


def model_abbr_from_cfg_used_in_summarizer(model):
    if model.get('summarizer_abbr', None):
        return model['summarizer_abbr']
    else:
        return model_abbr_from_cfg(model)

def post_process_single_rate(judgement: str):
    """Input a string like below:

    xxx[[5]]xxx, and extract the score
    """
    pattern = r'\[\[([\d.]+)\]\]'
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
    judge_model_abbr,
    dataset_abbr,
):
    capability_ratings = defaultdict(int)
    capability_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        capability_ratings['total'] += ans['score']
        capability_counts['total'] += 1
        capability_ratings[ref['capability']] += ans['score']
        capability_counts[ref['capability']] += 1

    capability_avg_ratings = defaultdict(float)

    for capability, total_score in capability_ratings.items():
        s = total_score / capability_counts[capability]
        s = round(s, 2)
        capability_avg_ratings[capability] = s
    columns = list(capability_avg_ratings.keys())
    columns.insert(0, columns.pop(columns.index('total')))

    if fout_flag == 0:
        with open(fout, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if fout_flag == 0:
                writer.writerow(['model', 'judge_model', 'dataset'] + columns)
            writer.writerow([model_abbr] + [judge_model_abbr] + [dataset_abbr] + [capability_avg_ratings[column] for column in columns])
    else:
        with open(fout, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([model_abbr] + [judge_model_abbr] + [dataset_abbr] + [capability_avg_ratings[column] for column in columns])
    return {column:capability_avg_ratings[column] for column in columns if column != ''}


class CommonSummarizer(CompassArenaSummarizer):
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='single_rate') -> None:
        self.judge_type = judge_type
        self.tasks = []
        self.cfg = config
        self.judge_type = 'single_rate'
        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.judge_model_cfgs = self.cfg['judge_models']
        self.judge_map = {
            'single_rate': post_process_single_rate
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
        fout_flag = 0
        output_tmp_file = osp.join(output_dir, 'result.csv')
        output_file = osp.join(output_dir, 'total_result.csv')
        json_result={}
        for eval_model_cfg in self.eval_model_cfgs:
            for judge_model_cfg in self.judge_model_cfgs:
                eval_model_abbr = model_abbr_from_cfg(eval_model_cfg)
                show_model_abbr = model_abbr_from_cfg_used_in_summarizer(eval_model_cfg)
                show_judge_model_abbr = model_abbr_from_cfg_used_in_summarizer(judge_model_cfg)
                judge_abbr = model_abbr_from_cfg(judge_model_cfg)
                subdir_path = os.path.join(results_folder, eval_model_abbr + '_judged-by--' + judge_abbr)
                if os.path.isdir(subdir_path):
                    for dataset in dataset_cfgs:
                        judged_answers, references = get_judgeanswer_and_reference(dataset, subdir_path, self.judge_function)
                        show_dataset_abbr = dataset_abbr_from_cfg(dataset)

                        tmp_result = get_capability_results(judged_answers, references, output_tmp_file, fout_flag, show_model_abbr, show_judge_model_abbr, show_dataset_abbr)
                        if show_judge_model_abbr not in json_result:
                            json_result[show_judge_model_abbr] = {}
                        json_result[show_judge_model_abbr][show_model_abbr] = tmp_result
                        fout_flag += 1
                else:
                    print(subdir_path + ' is not exist! please check!')
        with open(output_tmp_file, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            table = [line for line in csv_reader]

            new_header = [''] + [line[0] for line in table]
            new_table = [[h] + line[1:] for h, line in zip(header[1:], table)]
            new_table = [[h] + [line[i] for line in table] for i, h in enumerate(header[1:], start=1)]
            t = tabulate(new_table, headers=new_header)
        with open(output_file, 'a') as f:
            f.write(','.join(new_header) + '\n')
            for line in new_table:
                f.write(','.join(map(str, line)) + '\n')
            print(output_file)
        return {'qa_bench_' + show_dataset_abbr:json_result}
