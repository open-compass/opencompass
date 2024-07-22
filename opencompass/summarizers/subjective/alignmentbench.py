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

MAPPING = {
    '事实与解释型回答': ['事实正确性', '满足用户需求', '清晰度', '完备性'],
    '逻辑推理型回答': ['事实正确性', '满足用户需求', '逻辑连贯性', '完备性'],
    '生成型回答': ['事实正确性', '满足用户需求', '逻辑连贯性', '创造性', '丰富度'],
    '建议型回答': ['事实正确性', '满足用户需求', '公平与可负责程度', '创造性']
}


def detect_mapping(text):
    if '清晰度' in text and '完备性' in text:
        return '事实与解释型回答'
    elif '完备性' in text and '逻辑连贯性' in text:
        return '逻辑推理型回答'
    elif '创造性' in text and '丰富度' in text:
        return '生成型回答'
    elif '创造性' in text and '公平与可负责程度' in text:
        return '建议型回答'
    else:
        return None


def extract_missing_rating(text, search_type):
    searching_keys = MAPPING[search_type]
    result_dict = {}
    for k in searching_keys:
        matches = re.findall(rf'{k}.*?\n', text)
        result_dict[k] = None
        for match in reversed(matches):
            if re.findall(r'\d{1,2}', match):
                result_dict[k] = int(re.findall(r'\d{1,2}', match)[-1])
                break
    overall_number = re.findall('\d{1,2}', text)
    try:
        result_dict['综合得分'] = int(overall_number[-1])
    except:
        return {}
    return result_dict


def extract_rating_plus(text):
    pattern = r'{(.*?)}(?![^{]*{)'  # match last brackets
    match = re.search(pattern, text)

    if match:
        dictionary_str = match.group(1)
        kv_pattern = r"'(.*?)': (\d+)"
        matches = re.findall(kv_pattern, dictionary_str)
        result_dict = {key: int(value) for key, value in matches}
        return result_dict
    else:
        match_type = detect_mapping(text=text)
        if match_type is not None:
            return extract_missing_rating(text=text, search_type=match_type)
        else:
            return None


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


def post_process_alignbench_plus(judgement: str,
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
            try:
                return float(match.group(1))
            except ValueError:
                return -1
        return -1

    # judgement = judgement.replace('\n', '')
    rating = extract_rating_plus(judgement)

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
            try:
                return float(match.group(1))
            except ValueError:
                return -1
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
        s = total_score / dimension_counts[dimension]
        s = round(s, 2)
        dimension_avg_ratings[dimension] = s

    scores = {model: dimension_avg_ratings}
    rows = list(scores.keys())
    columns = list(scores[rows[0]].keys())
    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            writer.writerow(['模型'] + columns)

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
        s = total_score / capability_counts[capability]
        s = round(s, 2)
        capability_avg_ratings[capability] = s

    temp_list = []
    total_column_num = 2
    for category, sub_categories in categories.items():
        total_column_num += 1 + len(sub_categories)
        capability_avg_ratings[category + '总分'] = np.mean([
            np.mean(capability_avg_ratings[cat])
            for cat in categories[category]
        ])
        capability_avg_ratings[category + '总分'] = round(
            capability_avg_ratings[category + '总分'], 2)
        temp_list.append(category + '总分')
    capability_avg_ratings['总分'] = 0
    for temp in temp_list:
        capability_avg_ratings['总分'] += capability_avg_ratings[temp]
    capability_avg_ratings['总分'] /= len(temp_list)
    capability_avg_ratings['总分'] = round(capability_avg_ratings['总分'], 2)
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

        row = [model]
        row.append(scores[model]['总分'])
        for category, sub_categories in categories.items():
            row.append(scores[model][category + '总分'])
            for sub_category in sub_categories:
                row.append(scores[model][sub_category])
        writer.writerow(row)

    scores = scores[model]
    scores.pop('中文推理总分', None)
    scores.pop('中文语言总分', None)

    # Creating a new dictionary with '总分' as the first item
    updated_scores = {'总分': scores.pop('总分')}
    updated_scores.update(scores)
    return updated_scores


class AlignmentBenchSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='general') -> None:
        self.tasks = []
        self.cfg = config
        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.eval_model_abbrs = [
            model_abbr_from_cfg(model) for model in self.eval_model_cfgs
        ]
        self.judge_models = self.cfg.get('judge_models', None)
        self.judge_type = judge_type
        assert self.judge_type in [
            'general', 'autoj', 'judgelm', 'general_plus'
        ]
        self.judge_map = {
            'general': post_process_alignbench,
            'general_plus': post_process_alignbench_plus,
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
        all_scores = {}
        for judge_model in self.judge_models:
            score_by_judgemodel = {}
            judge_abbr = model_abbr_from_cfg(judge_model)
            dataset_cfgs = self.cfg['datasets']
            dataset = dataset_cfgs[0]  # Alignbench just have only one subfile
            output_dir, results_folder = get_outdir(self.cfg, time_str)
            fout_flag, fout_flag2 = 0, 0
            if self.judge_type == 'general':
                fout = osp.join(
                    output_dir,
                    'Alignbench-judged-by--' + judge_abbr + '-dimension.csv')
            fout2 = osp.join(
                output_dir,
                'Alignbench-judged-by--' + judge_abbr + '-capability.csv')

            for eval_model_abbr in self.eval_model_abbrs:
                subdir = eval_model_abbr + '_judged-by--' + judge_abbr
                subdir_path = os.path.join(results_folder, subdir)
                model = eval_model_abbr
                if os.path.isdir(subdir_path):
                    judged_answers, references = get_judgeanswer_and_reference(
                        dataset, subdir_path, self.judge_function)
                    if len(judged_answers) == 0:
                        score_by_judgemodel[model] = None
                        continue
                    if self.judge_type == 'general':
                        get_dimension_results(judged_answers, references, fout,
                                              fout_flag, model)
                        fout_flag += 1
                    scores = get_capability_results(judged_answers, references,
                                                    fout2, fout_flag2, model,
                                                    self.category)

                    score_by_judgemodel[model] = scores
                    fout_flag2 += 1
                else:
                    score_by_judgemodel[model] = None
                    print(subdir_path + ' is not exist! please check!')

            all_scores[judge_abbr] = score_by_judgemodel
        return {'Alignbench': all_scores}
