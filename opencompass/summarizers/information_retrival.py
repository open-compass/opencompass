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

from .subjective_post_process import post_process_autoj
from .utils import get_judgeanswer_and_reference, get_outdir


def post_process_ir(judgement: str):
    """Input a string like below:

    Conclusion: [[Correct]]\nReasoning: xxx
    and extract the score
    """
    matches = re.findall(r'\[\[(.*?)\]\]', judgement)
    if matches:
        matches = matches[0]
        if matches in ['Correct', 'Wrong', '对', '错']:
            if matches == 'Correct' or matches == '对':
                return {'score': 1}
            else:
                return {'score': 0}
        else:
            return None
    else:
        return None


def get_results(
    judged_answers,
    references,
    fout,
    fout_flag,
    model,
):
    capability_ratings = defaultdict(int)
    capability_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        lan = ref['others']['lan']
        capability_ratings['total'] += ans['score']
        capability_counts['total'] += 1
        capability_ratings[lan] += ans['score']
        capability_counts[lan] += 1

    capability_avg_ratings = defaultdict(float)

    for capability, total_score in capability_ratings.items():
        capability_avg_ratings[
            capability] = total_score / capability_counts[capability]

    scores = {model: capability_avg_ratings}

    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            num_header = [str(i) for i in range(4)]
            writer.writerow(num_header)

            header = ['模型']
            for category in capability_avg_ratings:
                header.append(category)
            writer.writerow(header)

        row = [model]
        for category in capability_avg_ratings:
            row.append(scores[model][category])
        writer.writerow(row)


class IRSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='autoj') -> None:
        self.tasks = []
        self.cfg = config
        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.eval_model_abbrs = [
            model_abbr_from_cfg(model) for model in self.eval_model_cfgs
        ]
        self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_model'])
        self.judge_type = judge_type
        assert self.judge_type in ['general', 'autoj']
        self.judge_map = {
            'general': post_process_ir,
            'autoj': post_process_autoj,
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
        fout_flag = 0
        for eval_model_abbr in self.eval_model_abbrs:
            subdir = eval_model_abbr + '_judged-by--' + self.judge_abbr
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                model, judge_model = eval_model_abbr, self.judge_abbr
                fout = osp.join(output_dir,
                                'judged-by--' + judge_model + '.csv')
                for dataset in dataset_cfgs:
                    judged_answers, references = get_judgeanswer_and_reference(
                        dataset, subdir_path, self.judge_function)
                    get_results(judged_answers, references, fout, fout_flag,
                                model)
                    fout_flag += 1
            else:
                print(subdir_path + ' is not exist! please check!')
        with open(fout, 'r') as f:
            x = from_csv(f)
        print(x)
