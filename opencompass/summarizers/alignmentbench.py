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


def post_process(judgment: str):

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

    def extract_score(text):
        pattern = r'\'综合得分\': (\d+(\.\d{1,2})?)'
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
        return -1

    def check_rating(rating):
        for k, v in rating.items():
            if isinstance(v, (int, float)) and ',' not in k:  # 确保值是数字
                pass
            else:
                return None
        return rating

    judgment = judgment.replace('\n', '')
    rating = extract_rating(judgment)

    if rating is not None:
        score = rating.get('综合得分', -1)
        if score == -1:
            score = extract_score(judgment)
        rating = check_rating(rating)
    else:
        score = -1
    return rating, score


class AlignmentBenchSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict) -> None:
        self.tasks = []
        self.cfg = config

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
        fout = osp.join(output_dir, 'report.csv')
        fout_flag = 0
        for subdir in os.listdir(results_folder):
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                model = subdir
                for dataset in dataset_cfgs:
                    dataset_abbr = dataset_abbr_from_cfg(dataset)
                    filepath = os.path.join(subdir_path,
                                            dataset_abbr + '.json')
                    result = mmengine.load(filepath)
                    judged_answers = []
                    references = []
                    for k, v in result.items():
                        rating, score = post_process(v['prediction'])
                        if rating is not None and score != -1:
                            judged_answers.append({
                                'rating': rating,
                                'score': score
                            })
                            references.append(v['gold'])
                    print(
                        f'Among {len(result)} judgements, successfully extracted {len(judged_answers)} judgements.'
                    )

                    # 初始化一个嵌套字典用于存储模型和评分
                    model_ratings = defaultdict(int)
                    model_counts = defaultdict(int)
                    for ans, ref in zip(judged_answers, references):
                        for k, v in ans['rating'].items():
                            if k != '综合得分':
                                model_ratings[k] += v
                                model_counts[k] += 1
                        model_ratings['综合得分'] += ans['score']
                        model_counts['综合得分'] += 1

                    model_avg_ratings = defaultdict(float)
                    for dimension, total_score in model_ratings.items():
                        model_avg_ratings[
                            dimension] = total_score / model_counts[dimension]

                    scores = {model: model_avg_ratings}
                    rows = list(scores.keys())
                    columns = list(scores[rows[0]].keys())
                    with open(fout, 'a+', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if fout_flag == 0:
                            writer.writerow([''] + columns)
                            fout_flag += 1
                        for row in rows:
                            writer.writerow(
                                [row] +
                                [scores[row][column] for column in columns])
        with open(fout, 'r') as f:
            x = from_csv(f)
        print(x)
