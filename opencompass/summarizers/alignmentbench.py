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
    '中文推理': ['数学计算', '逻辑推理'],
    '中文语言': ['基本任务', '中文理解', '综合问答', '文本写作', '角色扮演', '专业能力'],
}

all_dimensions = [
    '事实正确性', '满足用户需求', '安全无害', '清晰度', '逻辑性', '完备性', '创造性', '可负责程度', '逻辑连贯性',
    '公平与可负责程度', '丰富度', '综合得分'
]


def post_process(judgement: str):
    """Input a string like below:

    xxx{'事实正确性': 1, '满足用户需求': 1, '清晰度': 2, '完备性': 1, '综合得分': 1}xxx,
    and extract each score
    """

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

    def extract_score(text):
        pattern = r'\'综合得分\': (\d+(\.\d{1,2})?)'
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
        return -1

    def check_rating(rating):
        for k, v in rating.items():
            if isinstance(v, (int, float)) and k in all_dimensions:  # 确保值是数字
                if v >= 0 and v <= 10:
                    pass
                else:
                    return None
            else:
                return None
        return rating

    judgement = judgement.replace('\n', '')
    rating = extract_rating(judgement)

    if rating is not None:
        score = rating.get('综合得分', -1)
        if score == -1:
            score = extract_score(judgement)
        if score >= 0 and score <= 10:
            pass
        else:
            score = -1
        rating = check_rating(rating)
    else:
        score = -1
    if rating == None or score == -1:
        return None
    else:
        return {'rating': rating, 'score': score}


def post_process_autoj(judgement: str):
    """Input a string like below:

    xxx[[5]]xxx, and extract the score
    """
    pattern = r'\[(\d+)\]'
    matched_result = re.findall(pattern, judgement)
    if matched_result:
        score = int(matched_result[0])
    else:
        return None
    return {'score': score}


def post_process_judgelm(judgement: str):
    """Input a string like below:

    5, reason:xxx and extract the score
    """
    if len(judgement) >= 2:
        first_two_chars = judgement[:2]
        if first_two_chars.isdigit() and first_two_chars == '10':
            score = 10
        else:
            first_char = judgement[0]
            if first_char.isdigit() and 0 <= int(first_char) <= 9:
                score = int(first_char)
            else:
                return None
    elif len(judgement) == 1:
        if judgement.isdigit() and 0 <= int(judgement) <= 9:
            score = int(judgement)
        else:
            return None
    else:
        return None
    return {'score': score}


def get_dimension_results(judged_answers, references, fout, fout_flag, model):
    dimension_ratings = defaultdict(int)
    dimension_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        for k, v in ans['rating'].items():
            if k != '综合得分':
                dimension_ratings[k] += v
                dimension_counts[k] += 1
        dimension_ratings['综合得分'] += ans['score']
        dimension_counts['综合得分'] += 1

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


def get_capability_results(judged_answers, references, fout, fout_flag, model):
    capability_ratings = defaultdict(int)
    capability_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        capability_ratings[ref['capability']] += ans['score']
        capability_counts[ref['capability']] += 1

    capability_avg_ratings = defaultdict(float)

    for capability, total_score in capability_ratings.items():
        capability_avg_ratings[
            capability] = total_score / capability_counts[capability]

    capability_avg_ratings['中文推理总分'] = np.mean(
        [np.mean(capability_avg_ratings[cat]) for cat in CATEGORIES['中文推理']])
    capability_avg_ratings['中文语言总分'] = np.mean(
        [np.mean(capability_avg_ratings[cat]) for cat in CATEGORIES['中文语言']])
    capability_avg_ratings['总分'] = (capability_avg_ratings['中文推理总分'] +
                                    capability_avg_ratings['中文语言总分']) / 2

    scores = {model: capability_avg_ratings}
    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            num_header = [str(i) for i in range(12)]
            writer.writerow(num_header)

            header = ['模型', '总分']
            for category, sub_categories in CATEGORIES.items():
                header.append(category)
                header.extend([None for _ in range(len(sub_categories))])
            writer.writerow(header)

            sub_header = ['模型', '总分']
            for category, sub_categories in CATEGORIES.items():
                sub_header.extend([category + '总分'])
                sub_header.extend(sub_categories)
            writer.writerow(sub_header)
            fout_flag += 1

        row = [model]
        row.append(scores[model]['总分'])
        for category, sub_categories in CATEGORIES.items():
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

    def __init__(self, config: ConfigDict) -> None:
        self.tasks = []
        self.cfg = config
        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.eval_model_abbrs = [
            model_abbr_from_cfg(model) for model in self.eval_model_cfgs
        ]
        self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_model'])

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
                fout = osp.join(output_dir,
                                'judged-by--' + judge_model + '-dimension.csv')
                fout2 = osp.join(
                    output_dir,
                    'judged-by--' + judge_model + '-capability.csv')
                for dataset in dataset_cfgs:
                    judged_answers, references = get_judgeanswer_and_reference(
                        dataset, subdir_path, post_process)
                    get_dimension_results(judged_answers, references, fout,
                                          fout_flag, model)
                    get_capability_results(judged_answers, references, fout2,
                                           fout_flag2, model)
            else:
                print(subdir_path + ' is not exist! please check!')
        with open(fout, 'r') as f:
            x = from_csv(f)
        print(x)
        with open(fout2, 'r') as f:
            x = from_csv(f)
        print(x)


class AutojSummarizer(AlignmentBenchSummarizer):
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict) -> None:
        super().__init__(config)

    def summarize(self,
                  post_process=post_process_autoj,
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
                        dataset, subdir_path, post_process)
                    get_capability_results(judged_answers, references, fout,
                                           fout_flag, model)
            else:
                print(subdir_path + ' is not exist! please check!')
        with open(fout, 'r') as f:
            x = from_csv(f)
        print(x)


class JudgeLMSummarizer(AutojSummarizer):
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict) -> None:
        super().__init__(config)

    def summarize(self,
                  post_process=post_process_judgelm,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        super().summarize(post_process, time_str)
