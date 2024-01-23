# flake8: noqa: E501
import csv
import os
import os.path as osp
import re
import ast
from collections import defaultdict
from datetime import datetime
from prettytable import from_csv
from itertools import product

import mmengine
from mmengine import ConfigDict
from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg
from opencompass.partitioners.sub_naive import remove_duplicate_pairs

from .utils import get_judgeanswer_and_reference, get_outdir

def post_process_alpacav1(completion: str):
    r"""Parse a completion that contains a list of dictionary and returns the name of the preferred model.

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
        rank = [c for c in ordered_completions if c["model"] == 'model_1'][0]["rank"]
        if rank in [1, 2]:
            return {'rank': rank}
        else:
            return None
    except Exception as e:
        return None

class AlpacaSummarizerV1:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='general') -> None:
        self.tasks = []
        self.cfg = config
        self.base_models = self.cfg['eval']['partitioner']['base_models']
        self.compare_models = self.cfg['eval']['partitioner']['compare_models']
        self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_model'])
        self.judge_type = judge_type
        assert self.judge_type in ['general']
        self.judge_map = {
            'general': post_process_alpacav1,
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
        model_combinations = list(
            product(self.base_models, self.compare_models))
        unique_combinations = remove_duplicate_pairs(
            [combo for combo in model_combinations if combo[0] != combo[1]])

        for model_pair in unique_combinations:
            model1, model2, judge_model = model_pair[0]['abbr'], model_pair[1][
                'abbr'], self.judge_abbr
            subdir = model1 + '_' + model2 + '_judged-by--' + self.judge_abbr
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                fout = osp.join(output_dir,
                                'judged-by--' + judge_model + '-report.csv')
                for dataset in dataset_cfgs:
                    judged_answers, references = get_judgeanswer_and_reference(
                        dataset, subdir_path, self.judge_function)
                    win_model1, win_model2, categories = defaultdict(float), defaultdict(float), defaultdict(float)
                    model1, model2 = references[0]['answer1'], references[0]['answer2']
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
                        'win_' + model1: win_model1,
                        'win_' + model2: win_model2
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
            else:
                print(subdir_path + ' is not exist! please check!')
        with open(fout, 'r') as f:
            x = from_csv(f)
        print(x)
