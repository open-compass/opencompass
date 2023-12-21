# flake8: noqa: E501
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

import mmengine
from mmengine import ConfigDict

try:
    from prettytable import from_csv
except ImportError:
    from_csv = None

from opencompass.utils import dataset_abbr_from_cfg


def match_general_answer(s):
    temp = s[0]
    if temp in ['A', 'B', 'C', 'D']:
        return temp
    else:
        return None


def match_GPT4_answer(s):
    result = re.search(r'分数：(.)', s)
    if result:
        return int(result.group(1))
    else:
        return None


judge_map = {'smart': match_GPT4_answer, 'other': match_general_answer}


def call_function(name, arg):
    if name in judge_map:
        return judge_map[name](arg)
    else:
        print('Function not found in the map.')


class Creationv01Summarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, match_method='smart') -> None:
        self.tasks = []
        self.cfg = config
        self.match_method = match_method

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

        for subdir in os.listdir(results_folder):
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                model, judge_model = subdir.split('_')
                fout = osp.join(output_dir, judge_model + '-report.csv')
                for dataset in dataset_cfgs:
                    dataset_abbr = dataset_abbr_from_cfg(dataset)
                    filepath = os.path.join(subdir_path,
                                            dataset_abbr + '.json')
                    result = mmengine.load(filepath)
                    judged_answers = []
                    references = []
                    for k, v in result.items():
                        judged_answers.append(
                            call_function(self.match_method, v['prediction']))
                        references.append(v['gold'])
                    print(
                        f'Among {len(judged_answers)} judgements, successfully extracted {len(judged_answers)-judged_answers.count(None)} judgements.'
                    )
                    model_scores, categories = defaultdict(float), defaultdict(
                        float)
                    for prediction, reference in zip(judged_answers,
                                                     references):
                        categories[reference['capability']] += 1
                        if prediction is not None:
                            model_scores[reference['capability']] += prediction
                    for capability in categories:
                        if capability not in model_scores:
                            model_scores[capability] = 0.0
                        else:
                            model_scores[capability] = round(
                                model_scores[capability] /
                                categories[capability], 2)
                    scores = {model: model_scores}
                    rows = list(scores.keys())
                    columns = list(scores[rows[0]].keys())
                    with open(fout, 'a+', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([''] + columns)
                        for row in rows:
                            writer.writerow(
                                [row] +
                                [scores[row][column] for column in columns])
        with open(fout, 'r') as f:
            x = from_csv(f)
        print(x)
