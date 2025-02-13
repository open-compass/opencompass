# flake8: noqa
# yapf: disable
import csv
import os
import os.path as osp
import re
from collections import Counter, defaultdict
from datetime import datetime

from mmengine import ConfigDict

from opencompass.utils import model_abbr_from_cfg

from .compass_arena import model_abbr_from_cfg_used_in_summarizer
from .utils import get_judgeanswer_and_reference, get_outdir


def post_process_husimpleqa(judgement: str):
    pattern = r'\"evaluation\": \"(.*?)\"'
    matched_result = re.findall(pattern, judgement)
    try:
        judge = matched_result[0].lower()
        return {'judge': judge}
    except (ValueError, IndexError) as e:
        return None


def get_capability_results(
    judged_answers,
    references,
    fout,
    fout_flag,
    model_abbr,
):
    dim_judges = defaultdict(list)
    dim_counts = defaultdict(float)

    for ans, ref in zip(judged_answers, references):
        dim_judges['total'].append(ans)
        dim_counts['total'] += 1
        dim = ref['hu_specific_dim']
        dim_judges[dim].append(ans)
        dim_counts[dim] += 1

    col_name = ['model']
    column = [model_abbr]
    for dim, judges in dim_judges.items():
        c = Counter(judges)
        dim_count = dim_counts[dim]
        for judge in ['correct', 'incorrect', 'not_attempted']:
            count = c[judge]
            col_name.append(dim + ' ' + judge)
            column.append(round(count / dim_count, 2))
        col_name.append(dim + ' count')
        column.append(dim_count)

    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0:
            writer.writerow(col_name)
        writer.writerow(column)

class HuSimpleQASummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
    """

    def __init__(self, config: ConfigDict) -> None:
        self.judge_type = 'single'
        self.tasks = []
        self.cfg = config

        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.judge_abbr = model_abbr_from_cfg(self.cfg['judge_models'][0])
        self.judge_function = post_process_husimpleqa

    def summarize(self, time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """

        dataset_cfgs = self.cfg['datasets']
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        fout_flag = 0
        for eval_model_cfg in self.eval_model_cfgs:
            eval_model_abbr = model_abbr_from_cfg(eval_model_cfg)
            show_model_abbr = model_abbr_from_cfg_used_in_summarizer(eval_model_cfg)
            subdir_path = os.path.join(results_folder, eval_model_abbr + '_judged-by--' + self.judge_abbr)
            if os.path.isdir(subdir_path):
                fout = osp.join(output_dir, 'judged-by--' + self.judge_abbr + '-capability.csv')
                overall_judged_answers, overall_references = [], []
                for dataset in dataset_cfgs:
                    judged_answers, references = get_judgeanswer_and_reference(dataset, subdir_path, self.judge_function)
                    judged_answers = [item['judge'] for item in judged_answers]
                    overall_judged_answers += judged_answers
                    overall_references += references

                get_capability_results(
                    overall_judged_answers,
                    overall_references,
                    fout,
                    fout_flag,
                    show_model_abbr,
                )
                fout_flag += 1
            else:
                print(subdir_path + ' is not exist! please check!')
