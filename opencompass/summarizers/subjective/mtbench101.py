# flake8: noqa: E501
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmengine import ConfigDict

try:
    from prettytable import from_csv
except ImportError:
    from_csv = None

from opencompass.utils import model_abbr_from_cfg

from .compass_arena import CompassArenaSummarizer
from .utils import get_judgeanswer_and_reference, get_outdir

# from .utils.writer import Writer


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


def post_process_mtbench101(judgement: str):
    """Input a string like below:

    xxx[[5]]xxx, and extract the score
    """
    match = re.search(r'\[([0-9]+)\]', judgement)
    if match:
        score = int(match.group(1))

    else:
        return None

    return {'score': score, 'judgement': judgement}


def get_final_results(judged_answers, references, output_dir, fout_flag,
                      model):

    task_multi_id_scores = defaultdict(list)
    task_scores = defaultdict(list)

    for ans, ref in zip(judged_answers, references):

        task = ref['task']
        multi_id = ref['multi_id']
        score = ans['score']

        task_multi_id_scores[(task, multi_id)].append(score)

    for (task, multi_id), scores in task_multi_id_scores.items():
        min_score = min(scores)
        task_scores[task].append(min_score)

    final_task_scores = {
        task: sum(scores) / len(scores) if scores else 0
        for task, scores in task_scores.items()
    }

    fout = osp.join(output_dir, 'task_score.csv')

    columns = list(final_task_scores.keys())

    print('================task_score=====================')
    print(final_task_scores)

    with open(fout, 'a+', newline='') as csvfile:

        writer = csv.writer(csvfile)
        if fout_flag == 0:
            writer.writerow(['model'] + columns)
        writer.writerow([model] +
                        [final_task_scores[column] for column in columns])
    return 0


class MTBench101Summarizer(CompassArenaSummarizer):
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

        self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_models'][0])

        self.judge_function = post_process_mtbench101

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
        fout_flag = 0
        for eval_model_abbr in self.eval_model_abbrs:
            subdir = eval_model_abbr + '_judged-by--' + self.judge_abbr
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                model, judge_model = eval_model_abbr, self.judge_abbr

                for dataset in dataset_cfgs:
                    print()
                    judged_answers, references = get_judgeanswer_and_reference(
                        dataset, subdir_path, self.judge_function)
                    get_final_results(judged_answers, references, output_dir,
                                      fout_flag, model)
                    fout_flag += 1
            else:
                print(subdir_path + ' is not exist! please check!')
