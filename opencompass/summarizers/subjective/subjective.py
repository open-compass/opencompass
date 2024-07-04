# flake8: noqa: E501
import os.path as osp
from datetime import datetime

import pandas as pd
from mmengine import ConfigDict

from .utils import get_outdir


class SubjectiveSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, function: str) -> None:
        self.cfg = config
        self.function = function

    def summarize(
            self,
            subjective_scores: list,
            time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S'),
    ):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            subjective_scores (list of dicts): Container of saving score information for each datasets and models
            time_str (str): Timestamp for file naming.

        Returns:
            None
        """
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        # Create a DataFrame for each judgemodel
        judgemodel_dfs = {}
        for dataset in subjective_scores:
            for dataset_name, judgemodel_scores in dataset.items():
                for judgemodel_name, scores in judgemodel_scores.items():
                    if judgemodel_name not in judgemodel_dfs:
                        judgemodel_dfs[judgemodel_name] = pd.DataFrame(
                            columns=scores.keys())
                    judgemodel_dfs[judgemodel_name].loc[dataset_name] = list(
                        scores.values())

        # Save each DataFrame to a separate CSV file
        for judgemodel_name, df in judgemodel_dfs.items():
            fout = osp.join(
                output_dir, 'Subjective_all_results-judged-by--' +
                judgemodel_name + '.csv')
            print('Your subjective evaluation results have been saved at ' +
                  str(fout))
            df.to_csv(fout)
