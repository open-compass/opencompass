# flake8: noqa: E501
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

import mmengine
import numpy as np
from mmengine import ConfigDict

try:
    from prettytable import from_csv
except ImportError:
    from_csv = None

from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg


def extract_score(text):
    pattern = r'\'综合得分\': (\d+(\.\d{1,2})?)'
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return -1


class DongcaiSummarizer:
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

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        dataset_cfgs = self.cfg['datasets']
        work_dir = self.cfg['work_dir']
        self.work_dir = work_dir

        self.time_str = time_str
        output_path = osp.join(self.work_dir, 'summary',
                               f'summary_{self.time_str}.txt')
        output_dir = osp.join(osp.split(output_path)[0], f'{self.time_str}')
        mmengine.mkdir_or_exist(output_dir)
        results_folder = osp.join(work_dir, 'results')
        fout = osp.join(output_dir, 'dimension.csv')
        fout2 = osp.join(output_dir, 'capability.csv')
        fout_flag, fout_flag2 = 0, 0
        for subdir in os.listdir(results_folder):
            if subdir not in self.eval_model_abbrs:
                continue
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                model = subdir
                for dataset in dataset_cfgs:
                    dataset_abbr = dataset_abbr_from_cfg(dataset)
                    filepath = os.path.join(subdir_path,
                                            dataset_abbr + '.json')
                    result = mmengine.load(filepath)
                    judged_answers = []
                    references = []
                    for k, v in result.items():
                        score = extract_score(v['prediction'])
                        if score != -1:
                            judged_answers.append({'score': score})
                            references.append(v['gold'])
                    print(
                        f'Among {len(result)} judgements, successfully extracted {len(judged_answers)} judgements.'
                    )
        ###TODO Write your summarizer
