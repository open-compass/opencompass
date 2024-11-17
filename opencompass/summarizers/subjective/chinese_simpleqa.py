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
from .utils import get_outdir
from opencompass.utils import dataset_abbr_from_cfg
import mmengine

def post_process_csimpleqa(prediction):
    score = "C"
    try:
        match = re.search(r"(A|B|C)", prediction)
        score = match.group(0) if match else "C" 
    except:
        score = "C"
    return score

def get_judgeanswer_and_reference(dataset, subdir_path, post_process):
    """Extract judgements (scores) and references.

    Args:
        dataset (ConfigDict): Dataset config.
        subdir_path (str): Model path in results dir.
        post_process (function): The pre-defined extract function.
    """
    dataset_abbr = dataset_abbr_from_cfg(dataset)
    filename = osp.join(subdir_path, dataset_abbr + '.json')
    partial_filename = osp.join(subdir_path, dataset_abbr + '_0.json')
    if osp.exists(osp.realpath(filename)):
        result = mmengine.load(filename)
    elif osp.exists(osp.realpath(partial_filename)):
        filename = partial_filename
        result = {}
        i = 1
        partial_dict_flag = 0
        while osp.exists(osp.realpath(filename)):
            res = mmengine.load(filename)
            for k, v in res.items():
                result[partial_dict_flag] = v
                partial_dict_flag += 1
            filename = osp.join(subdir_path,
                                dataset_abbr + '_' + str(i) + '.json')
            i += 1
    else:
        result = {}

    if len(result) == 0:
        print('*' * 100)
        print('There are no results for ' + filename + ' or ' +
              partial_filename)
        print('*' * 100)

    judged_answers = []
    references = []
    result = result['details']
    for k, v in result.items():
        processed_judge = post_process(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            references.append(v['gold'])
    if len(judged_answers) <= 0.95 * len(result):
        print('*' * 100)
        print(
            f'For your {filename} judge. Among {len(result)} judgements, successfully extracted {len(judged_answers)} judgements, please check!'
        )
        print('*' * 100)
    return judged_answers, references

def calculate_metrics(judged_answers):
    # judged_answers is a list like ["A", "B", "C", ...]

    total_questions = len(judged_answers)
    total_correct = judged_answers.count("A")
    total_incorrect = judged_answers.count("B")
    total_not_attempted = judged_answers.count("C")
    
    total_correct_accuracy = total_correct / total_questions if total_questions > 0 else 0
    total_incorrect_accuracy = total_incorrect / total_questions if total_questions > 0 else 0
    total_not_attempted_accuracy = total_not_attempted / total_questions if total_questions > 0 else 0
    
    total_given_attempted_accuracy = total_correct / (total_correct + total_incorrect) if (total_correct + total_incorrect) > 0 else 0
    
    f1 = 2 * total_given_attempted_accuracy * total_correct_accuracy / (total_given_attempted_accuracy + total_correct_accuracy) if (total_given_attempted_accuracy + total_correct_accuracy) > 0 else 0
    
    return {
        'correct': total_correct_accuracy, 
        'incorrect': total_incorrect_accuracy, 
        'not_attempted': total_not_attempted_accuracy, 
        "given_attempted_accuracy": total_given_attempted_accuracy, 
        "F1": f1
    }
    
def get_results(judged_answers):
    results = calculate_metrics(judged_answers)
    return results

def get_dimension_results(judged_answers, references, fout, fout_flag, model):
    dimension_ratings = defaultdict(int)
    dimension_counts = defaultdict(int)

    results = get_results(judged_answers)
    f_score = results['F1']
    scores = {model: results}
    rows = list(scores.keys())
    columns = list(scores[rows[0]].keys())
    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            writer.writerow(['模型'] + columns)

        for row in rows:
            writer.writerow([row] +
                            [scores[row][column] for column in columns])
    return {
        "f_score": f_score
    }

def calculate_metrics(judged_answers):
    # judged_answers is a list like ["A", "B", "C", ...]

    total_questions = len(judged_answers)
    total_correct = judged_answers.count("A")
    total_incorrect = judged_answers.count("B")
    total_not_attempted = judged_answers.count("C")
    
    total_correct_accuracy = total_correct / total_questions if total_questions > 0 else 0
    total_incorrect_accuracy = total_incorrect / total_questions if total_questions > 0 else 0
    total_not_attempted_accuracy = total_not_attempted / total_questions if total_questions > 0 else 0
    
    total_given_attempted_accuracy = total_correct / (total_correct + total_incorrect) if (total_correct + total_incorrect) > 0 else 0
    
    f1 = 2 * total_given_attempted_accuracy * total_correct_accuracy / (total_given_attempted_accuracy + total_correct_accuracy) if (total_given_attempted_accuracy + total_correct_accuracy) > 0 else 0
    
    return {
        'correct': total_correct_accuracy, 
        'incorrect': total_incorrect_accuracy, 
        'not_attempted': total_not_attempted_accuracy, 
        "given_attempted_accuracy": total_given_attempted_accuracy, 
        "F1": f1
    }
    

class CsimpleqaSummarizer:
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
        self.judge_function = post_process_csimpleqa

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
            fout_flag = 0
           
            fout = osp.join(
                output_dir,
                'Chinesesimpleqa-judged-by--' + judge_abbr + '-capability.csv')

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

                    scores = get_dimension_results(judged_answers, references, fout,
                                              fout_flag, model)                              
                    fout_flag += 1
                    score_by_judgemodel[model] = scores
                else:
                    score_by_judgemodel[model] = None
                    print(subdir_path + ' is not exist! please check!')

            all_scores[judge_abbr] = score_by_judgemodel
        return {'Chinesesimpleqa': all_scores}
