import csv
import os
import os.path as osp
from datetime import datetime

import mmengine
from mmengine import ConfigDict
from prettytable import from_csv

from opencompass.utils import dataset_abbr_from_cfg


class SubjectSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(
        self,
        config: ConfigDict,
    ) -> None:
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
        for subdir in os.listdir(results_folder):
            subdir_path = os.path.join(results_folder, subdir)
            if os.path.isdir(subdir_path):
                for dataset in dataset_cfgs:
                    model1, model2 = dataset['eval_cfg']['evaluator'][
                        'base_model'], dataset['eval_cfg']['evaluator'][
                            'compare_model']
                    dataset_abbr = dataset_abbr_from_cfg(dataset)
                    filepath = os.path.join(subdir_path,
                                            dataset_abbr + '.json')
                    result = mmengine.load(filepath)
                    rows = list(result.keys())
                    columns = list(result[rows[0]].keys())
                    fout = osp.join(output_dir,
                                    model1 + '_vs_' + model2 + '.csv')
                    print(
                        '###############################Subjective Results on '
                        + model1 + '_vs_' + model2 +
                        '###############################')
                    with open(fout, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([model1 + '_vs_' + model2] + columns)
                        for row in rows:
                            writer.writerow(
                                [row] +
                                [result[row][column] for column in columns])
                    with open(fout, 'r') as f:
                        x = from_csv(f)
                    print(x)
