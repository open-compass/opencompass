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

from .subjective_post_process import post_process_autoj, post_process_judgelm
from .utils import get_judgeanswer_and_reference, get_outdir

CATEGORIES = {
    '中文推理': ['数学计算', '逻辑推理'],
    '中文语言': ['基本任务', '中文理解', '综合问答', '文本写作', '角色扮演', '专业能力'],
}

All_Dimensions = [
    '事实正确性', '满足用户需求', '安全无害', '清晰度', '逻辑性', '完备性', '创造性', '可负责程度', '逻辑连贯性',
    '公平与可负责程度', '丰富度', '综合得分'
]


def extract_rating(text):
    pattern = r'{(.*?)}(?![^{]*{)'  # match last brackets
    match = re.search(pattern, text)

    if match:
        dictionary_str = match.group(1)
        kv_pattern = r"'(.*?)': (\d+)"
        matches = re.findall(kv_pattern, dictionary_str)
        result_dict = {key: int(value) for key, value in matches}
        return result_dict
    else:
        return None


def check_rating(rating, all_dimensions):
    for k, v in rating.items():
        if isinstance(v, (int, float)) and k in all_dimensions:  # 确保值是数字
            if v >= 0 and v <= 10:
                pass
            else:
                return None
        else:
            return None
    return rating


def post_process_alignbench(judgement: str,
                            all_dimensions=All_Dimensions,
                            possible_keys=['综合得分']):
    """Input a string like below:

    xxx{'事实正确性': 1, '满足用户需求': 1, '清晰度': 2, '完备性': 1, '综合得分': 1}xxx,
    and extract each score
    """

    def extract_score(text):
        keys_pattern = '|'.join(map(re.escape, possible_keys))
        pattern = rf"({'|'.join(possible_keys)}): (\d+(\.\d{{1,2}})?)"
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
        return -1

    judgement = judgement.replace('\n', '')
    rating = extract_rating(judgement)

    if rating is not None:
        score = -1
        for key in possible_keys:
            score = rating.get(key, -1)
            if score != -1:
                break
        if score == -1:
            score = extract_score(judgement)
        if score >= 0 and score <= 10:
            pass
        else:
            score = -1
        rating = check_rating(rating, all_dimensions)
    else:
        score = -1
    if rating == None or score == -1:
        return None
    else:
        return {'rating': rating, 'score': score}


def get_dimension_results(judged_answers, references, fout, fout_flag, model):
    dimension_ratings = defaultdict(int)
    dimension_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        for k, v in ans['rating'].items():
            if k != '综合得分' or k != 'Overall Score':
                dimension_ratings[k] += v
                dimension_counts[k] += 1
            else:
                if k == '综合得分':
                    dimension_ratings['综合得分'] += ans['score']
                    dimension_counts['综合得分'] += 1
                else:
                    dimension_ratings['Overall Score'] += ans['score']
                    dimension_counts['Overall Score'] += 1

    dimension_avg_ratings = defaultdict(float)
    for dimension, total_score in dimension_ratings.items():
        dimension_avg_ratings[
            dimension] = total_score / dimension_counts[dimension]

    scores = {model: dimension_avg_ratings}
    rows = list(scores.keys())
    columns = list(scores[rows[0]].keys())
    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            writer.writerow(['模型'] + columns)
            fout_flag += 1
        for row in rows:
            writer.writerow([row] +
                            [scores[row][column] for column in columns])


def get_capability_results(judged_answers,
                           references,
                           fout,
                           fout_flag,
                           model,
                           categories=CATEGORIES):
    capability_ratings = defaultdict(int)
    capability_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        capability_ratings[ref['capability']] += ans['score']
        capability_counts[ref['capability']] += 1

    capability_avg_ratings = defaultdict(float)

    for capability, total_score in capability_ratings.items():
        capability_avg_ratings[
            capability] = total_score / capability_counts[capability]

    temp_list = []
    for category, sub_categories in categories.items():
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
            num_header = [str(i) for i in range(12)]
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


class AlignmentBenchSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type: str) -> None:
        self.tasks = []
        self.cfg = config
        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.eval_model_abbrs = [
            model_abbr_from_cfg(model) for model in self.eval_model_cfgs
        ]
        self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_model'])
        self.judge_type = judge_type
        assert self.judge_type in ['general', 'autoj', 'judgelm']
        self.judge_map = {
            'general': post_process_alignbench,
            'autoj': post_process_autoj,
            'judgelm': post_process_judgelm
        }
        self.judge_function = self.judge_map[self.judge_type]
        self.category = CATEGORIES

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
        fout_flag, fout_flag2 = 0, 0
        for eval_model_abbr in self.eval_model_abbrs:
            subdir = eval_model_abbr + '_judged-by--' + self.judge_abbr
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                model, judge_model = eval_model_abbr, self.judge_abbr
                if self.judge_type == 'general':
                    fout = osp.join(
                        output_dir,
                        'judged-by--' + judge_model + '-dimension.csv')
                fout2 = osp.join(
                    output_dir,
                    'judged-by--' + judge_model + '-capability.csv')
                for dataset in dataset_cfgs:
                    judged_answers, references = get_judgeanswer_and_reference(
                        dataset, subdir_path, self.judge_function)
                    if self.judge_type == 'general':
                        get_dimension_results(judged_answers, references, fout,
                                              fout_flag, model)
                    get_capability_results(judged_answers, references, fout2,
                                           fout_flag2, model, self.category)
            else:
                print(subdir_path + ' is not exist! please check!')
        if self.judge_type == 'general':
            with open(fout, 'r') as f:
                x = from_csv(f)
            print(x)
        with open(fout2, 'r') as f:
            x = from_csv(f)
        print(x)
