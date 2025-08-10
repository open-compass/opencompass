# flake8: noqa
# yapf: disable
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from mmengine import ConfigDict
from tabulate import tabulate

from opencompass.partitioners.sub_naive import remove_duplicate_pairs
from opencompass.summarizers.subjective.compass_arena import (
    check_position_bias, model_abbr_from_cfg_used_in_summarizer)
from opencompass.summarizers.subjective.utils import (
    get_judgeanswer_and_reference, get_outdir)
from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg


def post_process_wildbench_pair(judgement: str):
    pattern = r'\"choice\": \"(.*?)\"'
    matched_result = re.findall(pattern, judgement)
    if matched_result:
        return matched_result[0]
    else:
        return None

MAP = {
    'instruct': [
        '总分',
        '中文总分',
        '英文总分',
        'instruct/compassbench_2501_IF_en_chatIF_sub',
        'instruct/compassbench_2501_IF_en_functionalIF_sub',
        'instruct/compassbench_2501_IF_cn_chatIF_sub',
        'instruct/compassbench_2501_IF_cn_functionalIF_sub',
    ],
    'language': [
        '总分',
        '中文总分',
        '英文总分',
        'language/compassbench_v2501_language_zh_chat_sub',
        'language/compassbench_v2501_language_zh_nlp_sub',
        'language/compassbench_v2501_language_zh_creation_sub',
        'language/compassbench_v2501_language_en_chat_sub',
        'language/compassbench_v2501_language_en_nlp_sub',
        'language/compassbench_v2501_language_en_creation_sub',
    ],
    'code': [
        '总分',
        '中文总分',
        '英文总分',
        'code/compassbench_2501_code_arena_en_sub',
        'code/compassbench_2501_code_arena_zh_sub',
    ],
}


class CompassBenchSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, check_pos_bias=False) -> None:
        self.tasks = []
        self.cfg = config
        self.base_models = self.cfg['datasets'][0]['base_models']
        self.compare_models = self.cfg['eval']['partitioner']['models']
        self.judge_models = self.cfg.get('judge_models', None)
        self.meta_judge_model = self.cfg.eval.partitioner.get(
            'meta_judge_model', None)
        self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_models'][0])
        self.judge_function = post_process_wildbench_pair
        self.check_pos_bias = check_pos_bias

    def get_score(self, time_str):
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        model_combinations = list(
            product(self.base_models, self.compare_models))
        unique_combinations = remove_duplicate_pairs(
            [combo for combo in model_combinations if combo[0] != combo[1]])

        if self.meta_judge_model is not None:
            self.judge_models.append(self.meta_judge_model)

        scores = {}
        for idx, judge_model_cfg in enumerate(self.judge_models):
            judge_model = model_abbr_from_cfg(judge_model_cfg)
            scores[judge_model] = {}
            for dataset in self.cfg['datasets']:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                dataset_root, dataset_detail = (
                    dataset_abbr.split('/')[0],
                    dataset_abbr.split('/')[1],
                )
                scores[judge_model][dataset_abbr] = {}
                for model_pair in unique_combinations:
                    base_model = model_pair[0]['abbr']
                    compare_model = model_pair[1]['abbr']
                    if idx == len(self.judge_models):
                        subdir = (base_model + '_' + compare_model +
                                  '_summarized-by--' + judge_model)
                    else:
                        subdir = (base_model + '_' + compare_model +
                                  '_judged-by--' + judge_model)
                    subdir_path = os.path.join(results_folder, subdir)
                    if not os.path.isdir(subdir_path):
                        print(subdir_path + ' is not exist! please check!')
                        scores[judge_model][dataset_abbr][compare_model] = None
                        continue

                    judged_answers, references = get_judgeanswer_and_reference(
                        dataset, subdir_path, self.judge_function)
                    win_base_model = defaultdict(float)
                    win_compare_model = defaultdict(float)
                    score_mapping = {
                        'A++': 1,
                        'A+': 0.5,
                        'A=B': 0,
                        'B+': -0.5,
                        'B++': -1,
                    }
                    cnt = defaultdict(float)

                    for judged_answer, reference in zip(
                            judged_answers, references):
                        if judged_answer not in score_mapping:
                            continue
                        else:
                            flag = (1 if reference['answer1'] == base_model
                                    else -1)
                            score_1 = score_mapping[judged_answer] * flag
                            score_2 = -score_1

                            cnt[dataset_abbr] += 1
                            win_compare_model[dataset_abbr] += score_2
                            win_base_model[dataset_abbr] += score_1

                    for key, value in cnt.items():
                        win_base_model[key] = win_base_model[key] / value * 100
                        win_base_model[key] = round(win_base_model[key], 2)
                        win_compare_model[key] = (win_compare_model[key] /
                                                  value * 100)
                        win_compare_model[key] = round(win_compare_model[key],
                                                       2)

                    scores[judge_model][dataset_abbr][
                        compare_model] = win_compare_model

        return scores

    def summarize(
            self,
            time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S'),
    ):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        scores = self.get_score(time_str)
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        for judge_abbr, judge_scores in scores.items():
            new_score = {}
            for dataset_name, model_scores in judge_scores.items():
                dataset_root, dataset_detail = (
                    dataset_name.split('/')[0],
                    dataset_name.split('/')[1],
                )
                if dataset_root not in new_score:
                    new_score[dataset_root] = {}
                if '_en_' in dataset_detail:
                    for model_name, cate_score in model_scores.items():
                        if model_name not in new_score[dataset_root]:
                            new_score[dataset_root][model_name] = {}
                        if len(cate_score) == 0:
                            new_score[dataset_root][model_name]['英文总分'] = None
                        else:
                            new_score[dataset_root][model_name].update(
                                cate_score)
                            new_score[dataset_root][model_name]['英文总分'] = (
                                sum(cate_score.values()) / len(cate_score))
                elif '_cn_' in dataset_detail or '_zh_' in dataset_detail:
                    for model_name, cate_score in model_scores.items():
                        if model_name not in new_score[dataset_root]:
                            new_score[dataset_root][model_name] = {}
                        if len(cate_score) == 0:
                            new_score[dataset_root][model_name]['中文总分'] = None
                        else:
                            new_score[dataset_root][model_name].update(
                                cate_score)
                            new_score[dataset_root][model_name]['中文总分'] = (
                                sum(cate_score.values()) / len(cate_score))
            for dataset, models in new_score.items():
                for model, details in models.items():
                    if (details['英文总分'] is not None
                            and details['中文总分'] is not None):
                        average_score = (details['英文总分'] + details['中文总分']) / 2
                    else:
                        average_score = None
                    details['总分'] = average_score

            df = pd.DataFrame()
            # Iterate over the MAP and new_score to populate the DataFrame
            for category, headers in MAP.items():
                category_data = []
                for model, scores in new_score[category].items():
                    row_data = [model]
                    for header in headers:
                        # Append the score if available, otherwise append None
                        row_data.append(scores.get(header, None))
                    category_data.append(row_data)

                # Create a DataFrame for the category and concatenate with the main DataFrame
                new_headers = [category + '_' + item for item in headers]
                category_df = pd.DataFrame(category_data,
                                           columns=[category] + new_headers)
                df = pd.concat([df, category_df.set_index(category)], axis=1)

                df_transposed = df.T

            output_filename = osp.join(
                output_dir,
                'summarized-by--' + judge_abbr + '-' + '-report.csv',
            )

            transposed_csv_file_path = output_filename
            df_transposed.to_csv(transposed_csv_file_path)
            print(f'save to {output_filename}')
