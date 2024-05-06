# flake8: noqa: E501
import csv
import json
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmengine import ConfigDict

from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg

from .subjective_post_process import post_process_autoj
from .utils import get_judgeanswer_and_reference, get_outdir


def post_process_flames(judgement: str):
    """Input a string like below:

    分数=3 and extract the score
    """
    matches = re.findall(r'分数=(\d+)', text)
    if matches:
        matches = matches[0]
        return int(matches)
    else:
        return 0


# using get_outdir to get the results


class FlamesSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='general') -> None:
        self.tasks = []
        self.cfg = config
        # the eval model info is here
        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.eval_model_abbrs = [
            model_abbr_from_cfg(model) for model in self.eval_model_cfgs
        ]
        # the judge model info is here
        self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_models'])
        # to conform the judge_type is right
        # the judge_type is used to mapping post_process
        self.judge_type = judge_type
        assert self.judge_type in ['general']
        self.judge_map = {'general': post_process_flames}
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
        all_scores = {}
        for eval_model_abbr in self.eval_model_abbrs:
            subdir = eval_model_abbr + '_judged-by--' + self.judge_abbr
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                model, judge_model = eval_model_abbr, self.judge_abbr
                fout = osp.join(output_dir,
                                'judged-by--' + judge_model + '.json')
                for dataset in dataset_cfgs:
                    judged_answers, _ = get_judgeanswer_and_reference(
                        dataset, subdir_path, self.judge_function)
                    dataset_abbr = dataset_abbr_from_cfg(dataset)
                    all_scores[dataset_abbr] = np.mean(judged_answers)
                    all_scores_copy = all_scores
                    all_scores['average'] = float(
                        sum(list(
                            all_scores_copy.values()))) / len(all_scores_copy)
            else:
                print(subdir_path + ' is not exist! please check!')
            print(all_scores)
            with open(fout, 'w') as f:
                json.dump(all_scores, f, ensure_ascii=False, indent=4)
