# flake8: noqa
# yapf: disable
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime
from itertools import product

import pandas as pd
from mmengine import ConfigDict

from opencompass.partitioners.sub_naive import remove_duplicate_pairs
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



class QaCompassBenchSummarizer:
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
                            cnt[reference['category']] += 1
                            win_compare_model[reference['category']] += score_2
                            win_base_model[reference['category']] += score_1
                            cnt[dataset_abbr] += 1
                            win_compare_model[dataset_abbr] += score_2
                            win_base_model[dataset_abbr] += score_1
                    for key, value in cnt.items():
                        # print(key , value)
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
        json_result={}
        for judge_abbr, judge_scores in scores.items():
            if judge_abbr not in json_result:
                json_result[judge_abbr] = {}
            new_score = {}
            items = []
            for dataset_name, model_scores in judge_scores.items():
                if dataset_name not in new_score:
                    new_score[dataset_name] = {}
                for model_name, cate_score in model_scores.items():
                    for category, score in cate_score.items():
                        items.append(category)
                        if category not in new_score:
                            new_score[category] = {}
                        if model_name not in new_score[category]:
                            new_score[category][model_name] = {}
                        new_score[category][model_name]['总分'] = score
                        if model_name not in json_result[judge_abbr]:
                            json_result[judge_abbr][model_name] = {}
                        json_result[judge_abbr][model_name][category] = score

            df = pd.DataFrame()
            # Iterate over the MAP and new_score to populate the DataFrame
            for category in items:
                category_data = []
                for model, scores in new_score[category].items():
                    row_data = [model]
                    # Append the score if available, otherwise append None
                    row_data.append(scores.get('总分', None))
                    category_data.append(row_data)

                # Create a DataFrame for the category and concatenate with the main DataFrame
                new_headers = [category + '_' + item for item in ['总分']]
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
            return {'qabench': json_result}
