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

from .utils import get_judgeanswer_and_reference, get_outdir

CATEGORIES = {
    '中文': ['json_zh', 'csv_zh', 'email_zh', 'markdown_zh', 'article_zh'],
    '英文': ['json_en', 'csv_en', 'email_en', 'markdown_en', 'article_en'],
}


def post_process_multiround(judgement: str):
    """Input a string like below:

    xxx输出：[1, 2, 3, 4, 5, 6]xxx,
    xxxOutput: [1, 2, 3, 4, 5, 6]xxx,
    and extract the list
    """
    pattern = r'\[([^]]*)\]'
    match = re.search(pattern, judgement)
    if match:
        temp = match.group(1)
        if temp == '':
            return 0
        numbers = temp.split(', ')
        try:
            if all(num.isdigit() for num in numbers):
                return len([int(num) for num in numbers])
            else:
                return None
        except ValueError:
            return None
    else:
        return None


def get_capability_results(judged_answers,
                           references,
                           fout,
                           fout_flag,
                           model,
                           categories=CATEGORIES):
    capability_ratings = defaultdict(float)
    capability_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        lan = ref['others']['language']
        capability_ratings[ref['capability'] + '_' +
                           lan] += (ref['others']['round'] -
                                    ans) / ref['others']['round']
        capability_counts[ref['capability'] + '_' + lan] += 1

    capability_avg_ratings = defaultdict(float)

    for capability, total_score in capability_ratings.items():
        capability_avg_ratings[
            capability] = total_score / capability_counts[capability]

    temp_list = []
    total_column_num = 2
    for category, sub_categories in categories.items():
        total_column_num += 1 + len(sub_categories)
        capability_avg_ratings[category + '总分'] = np.mean([
            np.mean(capability_avg_ratings[cat])
            for cat in categories[category]
        ])
        temp_list.append(category + '总分')
    capability_avg_ratings['总分'] = 0
    for temp in temp_list:
        capability_avg_ratings['总分'] += capability_avg_ratings[temp]
    capability_avg_ratings['总分'] /= len(temp_list)
    scores = {model: capability_avg_ratings}

    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            num_header = [str(i) for i in range(total_column_num)]
            writer.writerow(num_header)

            header = ['模型', '总分']
            for category, sub_categories in categories.items():
                header.append(category)
                header.extend([None for _ in range(len(sub_categories))])
            writer.writerow(header)

            sub_header = ['模型', '总分']
            for category, sub_categories in categories.items():
                sub_header.extend([category + '总分'])
                sub_header.extend(sub_categories)
            writer.writerow(sub_header)
            fout_flag += 1

        row = [model]
        row.append(scores[model]['总分'])
        for category, sub_categories in categories.items():
            row.append(scores[model][category + '总分'])
            for sub_category in sub_categories:
                row.append(scores[model][sub_category])
        writer.writerow(row)


class MultiroundSummarizer:
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
        self.judge_abbr = model_abbr_from_cfg(
            self.cfg['eval']['partitioner']['judge_models'][0])

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
                fout = osp.join(
                    output_dir,
                    'judged-by--' + judge_model + '-capability.csv')
                for dataset in dataset_cfgs:
                    judged_answers, references = get_judgeanswer_and_reference(
                        dataset, subdir_path, post_process_multiround)
                    get_capability_results(judged_answers, references, fout,
                                           fout_flag, model)
            else:
                print(subdir_path + ' is not exist! please check!')
        with open(fout, 'r') as f:
            x = from_csv(f)
        print(x)
