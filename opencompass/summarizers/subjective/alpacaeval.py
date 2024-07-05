# flake8: noqa: E501
import ast
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime
from itertools import product

import mmengine
from mmengine import ConfigDict
from prettytable import from_csv

from opencompass.partitioners.sub_naive import remove_duplicate_pairs
from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg

from .utils import get_judgeanswer_and_reference, get_outdir


def post_process_alpacav1(completion: str):
    r"""Parse a completion that contains a list of dictionary and returns the rank of the model1.

    Examples
    --------
    >>> ranking_parser("[{'model': 'model_1', 'rank': 1}, {'model': 'model_2', 'rank': 2}]")
    1
    >>> ranking_parser("[{'model': 'model_1', 'rank': 2}, {'model': 'model_2', 'rank': 1}]")
    2
    >>> ranking_parser("[{'model': 'model_1', 'rank': 3}, {'model': 'model_2', 'rank': 1}]")
    None
    """
    try:
        if isinstance(completion, str):
            completion = re.findall(r'\[.*?\]', completion)[0]
            ordered_completions = ast.literal_eval(completion)
        else:
            ordered_completions = completion
        rank = [c for c in ordered_completions
                if c['model'] == 'model_1'][0]['rank']
        if rank in [1, 2]:
            return {'rank': rank}
        else:
            return None
    except Exception as e:
        return None


def post_process_alpacav2(completion: str):
    r"""Parse a completion that contains 'm' or 'M' and returns the rank of the model1.

    Examples
    --------
    >>> ranking_parser("m")
    1
    >>> ranking_parser("M")
    2
    >>> ranking_parser("s")
    None
    """
    try:
        if completion[0] == 'm':
            return {'rank': 1}
        elif completion[0] == 'M':
            return {'rank': 2}
        else:
            return None
    except Exception as e:
        return None


class AlpacaSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='v2') -> None:
        self.tasks = []
        self.cfg = config
        self.base_models = self.cfg['datasets'][0]['base_models']
        self.compare_models = self.cfg['eval']['partitioner']['models']
        self.judge_models = self.cfg.get('judge_models', None)
        self.judge_type = judge_type
        assert self.judge_type in ['v1', 'v2']
        self.judge_map = {
            'v1': post_process_alpacav1,
            'v2': post_process_alpacav2
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
        all_scores = {}
        for judge_model in self.judge_models:
            score_by_judgemodel = {}
            judge_abbr = model_abbr_from_cfg(judge_model)
            dataset_cfgs = self.cfg['datasets']
            dataset = dataset_cfgs[0]  # AlpacaEval just have only one subfile
            dataset_abbr = dataset_abbr_from_cfg(dataset)
            output_dir, results_folder = get_outdir(self.cfg, time_str)
            model_combinations = list(
                product(self.base_models, self.compare_models))
            unique_combinations = remove_duplicate_pairs([
                combo for combo in model_combinations if combo[0] != combo[1]
            ])

            for model_pair in unique_combinations:
                model1, model2 = model_pair[0]['abbr'], model_pair[1]['abbr']
                subdir = model1 + '_' + model2 + '_judged-by--' + judge_abbr
                subdir_path = os.path.join(results_folder, subdir)
                filename = osp.realpath(
                    osp.join(subdir_path, dataset_abbr + '.json'))
                partial_filename = osp.realpath(
                    osp.join(subdir_path, dataset_abbr + '_0.json'))
                if osp.exists(osp.realpath(filename)) or osp.exists(
                        osp.realpath(partial_filename)):
                    fout = osp.join(
                        output_dir,
                        'AlpacaEval2-judged-by--' + judge_abbr + '.csv')

                    judged_answers, references = get_judgeanswer_and_reference(
                        dataset, subdir_path, self.judge_function)
                    win_model1, win_model2, categories = defaultdict(
                        float), defaultdict(float), defaultdict(float)
                    model1, model2 = references[0]['answer1'], references[0][
                        'answer2']
                    for prediction, reference in zip(judged_answers,
                                                     references):
                        categories['total'] += 1
                        categories[reference['capability']] += 1
                        if prediction['rank'] == 1:
                            if reference['answer1'] == model1:
                                win_model1[reference['capability']] += 1
                                win_model1['total'] += 1
                            else:
                                win_model2[reference['capability']] += 1
                                win_model2['total'] += 1
                        else:
                            if reference['answer1'] == model1:
                                win_model2[reference['capability']] += 1
                                win_model2['total'] += 1
                            else:
                                win_model1[reference['capability']] += 1
                                win_model1['total'] += 1
                    for capability in categories:
                        if capability not in win_model1:
                            win_model1[capability] = 0.0
                        else:
                            win_model1[capability] = round(
                                (win_model1[capability] /
                                 categories[capability]) * 100, 2)
                        if capability not in win_model2:
                            win_model2[capability] = 0.0
                        else:
                            win_model2[capability] = round(
                                (win_model2[capability] /
                                 categories[capability]) * 100, 2)

                    scores = {
                        #'win_' + model1: win_model1, # We just show winrate of model2, because model1 is base model and only one model as base model in AlpacaEval
                        'win_' + model2:
                        win_model2
                    }
                    rows = list(scores.keys())
                    columns = list(scores[rows[0]].keys())
                    columns.insert(0, columns.pop(columns.index('total')))
                    with open(fout, 'a+', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([model1 + '_vs_' + model2] + columns)
                        for row in rows:
                            writer.writerow(
                                [row] +
                                [scores[row][column] for column in columns])
                    win_model2_update = {'total': win_model2.pop('total')}
                    win_model2_update.update(win_model2)
                    score_by_judgemodel[model2] = win_model2_update
                else:
                    score_by_judgemodel[model2] = None
                    # print(subdir_path + ' is not exist! please check!')
            all_scores[judge_abbr] = score_by_judgemodel
        return {'AlpacaEval': all_scores}
