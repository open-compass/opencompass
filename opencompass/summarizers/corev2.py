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
    if result := re.findall('(?:选择：|Choice: )([ABCD])', s):
        return result[0]
    else:
        return None

judge_map = {'gpt4': match_GPT4_answer, 'other': match_general_answer}

def call_function(name, arg):
    if name in judge_map:
        return judge_map[name](arg)
    else:
        print('Function not found in the map.')


class Corev2Summarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(
        self,
        config: ConfigDict,
        judge_method='gpt4'
    ) -> None:
        self.tasks = []
        self.cfg = config
        self.judge_method = judge_method

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
        fout = osp.join(output_dir,'report.csv')
        for subdir in os.listdir(results_folder):
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                model1, model2 = subdir.split('_')
                for dataset in dataset_cfgs:
                    dataset_abbr = dataset_abbr_from_cfg(dataset)
                    filepath = os.path.join(subdir_path,
                                            dataset_abbr + '.json')
                    result = mmengine.load(filepath)
                    judged_answers = []
                    references = []
                    for k, v in result.items():
                        judged_answers.append(call_function(self.judge_method, v['prediction']))
                        references.append(v['gold'])
                    print(
                        f'Among {len(judged_answers)} judgements, successfully extracted {len(judged_answers)-judged_answers.count(None)} judgements.'
                    )
                    win_both_model1, win_both_model2, half_draw_model1, half_draw_model2, categories = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
                    model1 = references[0]['answer1']
                    model2 = references[0]['answer2']
                    for prediction, reference in zip(judged_answers, references):
                        if prediction is not None:
                            categories[reference['capability'].split('-')[0]] += 1
                            categories[reference['capability']] += 1
                            winner = ''
                            if prediction == 'A':
                                winner = reference['answer1']
                            elif prediction == 'B':
                                winner = reference['answer2']
                            elif prediction == 'C':
                                win_both_model1[reference['capability'].split('-')[0]] += 1
                                win_both_model2[reference['capability'].split('-')[0]] += 1
                                win_both_model1[reference['capability']] += 1
                                win_both_model2[reference['capability']] += 1
                            if model1 == winner:
                                half_draw_model1[reference['capability'].split('-')[0]] += 1
                                win_both_model1[reference['capability'].split('-')[0]] += 1
                                half_draw_model1[reference['capability']] += 1
                                win_both_model1[reference['capability']] += 1
                            elif model2 == winner:
                                half_draw_model2[reference['capability'].split('-')[0]] += 1
                                win_both_model2[reference['capability'].split('-')[0]] += 1 
                                half_draw_model2[reference['capability']] += 1
                                win_both_model2[reference['capability']] += 1 
                    for capability in categories:
                        if capability not in half_draw_model1:
                            win_both_model1[capability] = 0.0
                            half_draw_model1[capability] = 0.0
                        else:
                            win_both_model1[capability] = round(
                                (win_both_model1[capability] / categories[capability]) * 100, 2)
                            half_draw_model1[capability] = round(
                                (half_draw_model1[capability] / categories[capability]) * 100, 2)
                        if capability not in half_draw_model2:
                            win_both_model2[capability] = 0.0
                            half_draw_model2[capability] = 0.0
                        else:
                            win_both_model2[capability] = round(
                                (win_both_model2[capability] / categories[capability]) * 100, 2)
                            half_draw_model2[capability] = round(
                                (half_draw_model2[capability] / categories[capability]) * 100, 2)
                    scores = {'win_both_'+model1: win_both_model1, 'half_draw_'+model1: half_draw_model1, 'win_both_'+model2: win_both_model2, 'half_draw_'+model2: half_draw_model2}
                    rows = list(scores.keys())
                    columns = list(scores[rows[0]].keys())
                    with open(fout, 'a+', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([model1+'_vs_'+model2] + columns)
                        for row in rows:
                            writer.writerow(
                                [row] +
                                [scores[row][column] for column in columns])
        with open(fout, 'r') as f:
            x = from_csv(f)
        print(x)
