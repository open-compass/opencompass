# flake8: noqa: E501
import csv
import json
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

import mmengine
import numpy as np
import pandas as pd
from mmengine import ConfigDict
from prettytable import from_csv

from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               model_abbr_from_cfg)

from .utils import get_outdir


def post_process_charm_mem(judgement: str):
    """Input a string like below:

    xxx[correct]xxx, and extract the judge
    """
    pattern = r'(?i)\[(incorrect|correct|正确|错误|Yes|No)\]'
    matched_result = re.findall(pattern, judgement)
    if matched_result:
        content = matched_result[0].lower()
        if content in ['correct', '正确', 'yes']:
            return {'correct': True}
        elif content in ['incorrect', '错误', 'no']:
            return {'correct': False}
    else:
        return None


def get_judgeanswer_and_reference_charm_mem(dataset, subdir_path,
                                            post_process):
    """Extract judgements (scores), references and original judging prompts.

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
        assert len(result) > 0

    judging_prompts = []
    judged_answers = []
    references = []
    for k, v in result.items():
        processed_judge = post_process(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            references.append(v['gold'])
            judging_origin_prompts = v['origin_prompt']
            if len(judging_origin_prompts) > 0:
                judging_prompts.append(judging_origin_prompts[0].get(
                    'prompt', None))
    if len(judged_answers) != len(result):
        print(
            f'Among {len(result)} judgements, successfully extracted {len(judged_answers)} judgements, please check!'
        )
    if len(judged_answers) == 0:
        print('*' * 100)
        print(
            'There are no extracted judgements, please change your judge model or check your prompt!!!'
        )
        print('*' * 100)
    assert len(judged_answers) > 0
    return judged_answers, references, judging_prompts


def get_accuracy(judged_answers):
    n_total = 0
    n_correct = 0
    for ans in judged_answers:
        if ans.get('correct', False):
            n_correct += 1
        n_total += 1

    return round(n_correct / n_total * 100, 2)


class CharmMemSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='single') -> None:
        self.judge_type = judge_type
        self.tasks = []
        self.cfg = config
        if self.judge_type == 'single':
            self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
            self.eval_model_abbrs = [
                model_abbr_from_cfg(model) for model in self.eval_model_cfgs
            ]
        else:
            raise NotImplementedError

        self.judge_abbr = model_abbr_from_cfg(
            self.cfg['eval']['partitioner']['judge_models'][0])
        self.judge_map = {'single': post_process_charm_mem}
        self.judge_function = self.judge_map[self.judge_type]

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        if self.judge_type == 'single':
            dataset_cfgs = self.cfg['datasets']
            judge_model = self.judge_abbr
            output_dir, results_folder = get_outdir(self.cfg, time_str)

            accuracy_df = pd.DataFrame(columns=self.eval_model_abbrs)
            for dataset in dataset_cfgs:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                dataset_instance = build_dataset_from_cfg(dataset)
                out_dir = osp.join(
                    output_dir,
                    'judged-by--' + judge_model + '-' + dataset_abbr)
                os.makedirs(out_dir, exist_ok=True)

                cur_acc_dict = {'dataset': dataset_abbr}
                for eval_model_abbr in self.eval_model_abbrs:
                    subdir = eval_model_abbr + '_judged-by--' + self.judge_abbr
                    subdir_path = os.path.join(results_folder, subdir)
                    if os.path.isdir(subdir_path):
                        model = eval_model_abbr
                        (judged_answers, references, judging_prompts
                         ) = get_judgeanswer_and_reference_charm_mem(
                             dataset,
                             subdir_path,
                             self.judge_function,
                         )
                        accuracy = get_accuracy(judged_answers)
                        cur_acc_dict[eval_model_abbr] = accuracy

                        detail_dict = {}
                        for i in range(len(judged_answers)):
                            cur_dict = {}
                            cur_dict['judging_prompt'] = judging_prompts[i]
                            for input_col in dataset_instance.reader.input_columns:
                                cur_dict[input_col] = dataset_instance.reader[
                                    'test'][input_col][i]
                            cur_dict['reference'] = references[i]
                            cur_dict.update(judged_answers[i])

                            detail_dict[str(i)] = cur_dict

                        out_dict = {'score': accuracy, 'details': detail_dict}
                        fout = osp.join(out_dir, model + '.json')
                        with open(fout, 'w', encoding='utf-8') as f:
                            json.dump(out_dict,
                                      f,
                                      indent=4,
                                      ensure_ascii=False)
                    else:
                        print(subdir_path + ' is not exist! please check!')

                accuracy_df = accuracy_df.append(cur_acc_dict,
                                                 ignore_index=True)
            accuracy_df.set_index('dataset', inplace=True)

            accuracy_file = osp.join(output_dir,
                                     'judged-by--' + judge_model + '.csv')
            accuracy_df.to_csv(accuracy_file, index=True)
            with open(accuracy_file, 'r') as f:
                x = from_csv(f)
            print(x)
