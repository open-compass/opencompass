# flake8: noqa: E501
import os.path as osp
from datetime import datetime

import pandas as pd
from mmengine import ConfigDict

from .utils import get_outdir


# Flatten the nested structure and ensure consistent order of models across datasets
def flatten_data(data):
    flat_data = {}
    models_order = set()
    for dataset in data:
        for dataset_name, judgemodel_scores in dataset.items():
            for judgemodel_name, model_scores in judgemodel_scores.items():
                if judgemodel_name not in flat_data:
                    flat_data[judgemodel_name] = {}
                if dataset_name not in flat_data[judgemodel_name]:
                    flat_data[judgemodel_name][dataset_name] = {}
                for model_name, scores in model_scores.items():
                    models_order.add(model_name)
                    if scores is not None:
                        for score_name, score_value in scores.items():
                            flat_data[
                                judgemodel_name][dataset_name].setdefault(
                                    score_name,
                                    {}).setdefault(model_name, score_value)
                    else:
                        for score_name in flat_data[judgemodel_name][
                                dataset_name]:
                            flat_data[judgemodel_name][dataset_name][
                                score_name].setdefault(model_name, None)

    # Ensure consistent order of models
    consistent_models_order = sorted(list(models_order))

    for judgemodel_name in flat_data:
        for dataset_name in flat_data[judgemodel_name]:
            for score_name in flat_data[judgemodel_name][dataset_name]:
                for model_name in consistent_models_order:
                    flat_data[judgemodel_name][dataset_name][
                        score_name].setdefault(model_name, None)

    return flat_data, consistent_models_order


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

        flat_data, models_order = flatten_data(subjective_scores)

        # Create a DataFrame for each judgemodel with models as rows and datasets as columns
        judgemodel_dfs_final_corrected = {}
        for judgemodel_name, datasets_scores in flat_data.items():
            dfs = {}  # Dictionary to hold DataFrames for each dataset
            for dataset_name, scores in datasets_scores.items():
                # Create a DataFrame with models as index and datasets as columns
                df = pd.DataFrame.from_dict(scores,
                                            orient='index',
                                            columns=models_order)
                # Insert a new row at the top for the dataset names
                df.insert(0, 'Detailed Scores', df.index.values)
                df.insert(0, 'Dataset',
                          [dataset_name for _ in range(len(df.index))])
                dfs[dataset_name] = df

            # Concatenate all DataFrames for the current judgemodel
            judgemodel_df = pd.concat(dfs.values(), ignore_index=True)
            judgemodel_dfs_final_corrected[judgemodel_name] = judgemodel_df

        # Save each DataFrame to a separate CSV file
        for judgemodel_name, df in judgemodel_dfs_final_corrected.items():
            fout = osp.join(
                output_dir, 'Subjective_all_results-judged-by--' +
                judgemodel_name + '.csv')
            print('Your subjective evaluation results have been saved at ' +
                  str(fout))
            df.to_csv(fout, index=False)
