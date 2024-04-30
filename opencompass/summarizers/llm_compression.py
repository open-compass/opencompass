import getpass
import os.path as osp
from datetime import datetime
from typing import List, Optional

import mmengine
import pandas as pd
from mmengine import ConfigDict

from opencompass.utils import dataset_abbr_from_cfg
from opencompass.utils.prompt import get_prompt_hash

from .default import DefaultSummarizer


class LLMCompressionSummarizer(DefaultSummarizer):

    def __init__(self,
                 config: ConfigDict,
                 dataset_abbrs: Optional[List[str]] = None,
                 summary_groups: List = None,
                 prompt_db=None) -> None:

        summary_groups = [] if summary_groups is None else summary_groups
        super().__init__(config, dataset_abbrs, summary_groups, prompt_db)

    def _format_table(self, parsed_results, dataset_metrics,
                      dataset_eval_mode):
        dataset_abbrs = [
            dataset_abbr_from_cfg(dataset) for dataset in self.dataset_cfgs
        ]
        prompt_version = {
            dataset_abbr_from_cfg(d): get_prompt_hash(d)[:6]
            for d in self.dataset_cfgs
        }

        summarizer_dataset_abbrs = []
        if self.dataset_abbrs is None:
            # display all dataset metrics included in the config
            for dataset_abbr in dataset_abbrs:
                if dataset_abbr in dataset_metrics:
                    for metric in dataset_metrics[dataset_abbr]:
                        summarizer_dataset_abbrs.append((dataset_abbr, metric))
                else:
                    summarizer_dataset_abbrs.append((dataset_abbr, None))
            # along with all possible group metrics
            for dataset_abbr in dataset_metrics:
                for metric in dataset_metrics[dataset_abbr]:
                    if (dataset_abbr, metric) not in summarizer_dataset_abbrs:
                        summarizer_dataset_abbrs.append((dataset_abbr, metric))
        else:
            # follow the required order
            for item in self.dataset_abbrs:
                if isinstance(item, str):
                    summarizer_dataset_abbrs.append((item, None))
                elif isinstance(item, (list, tuple)):
                    summarizer_dataset_abbrs.append((item[0], item[1]))

        table = []
        header = ['dataset', 'version', 'metric', 'mode'] + self.model_abbrs
        table.append(header)
        for dataset_abbr, metric in summarizer_dataset_abbrs:
            if dataset_abbr not in dataset_metrics:
                table.append([dataset_abbr, '-', '-', '-'] +
                             ['-'] * len(self.model_abbrs))
                continue
            if metric is None:
                metric = dataset_metrics[dataset_abbr][0]
            elif metric in dataset_metrics[dataset_abbr]:
                pass
            else:
                table.append([dataset_abbr, '-', '-', '-'] +
                             ['-'] * len(self.model_abbrs))
                continue

            row = [
                dataset_abbr,
                prompt_version.get(dataset_abbr, '-'), metric,
                dataset_eval_mode.get(dataset_abbr, '-')
            ]
            for model_abbr in self.model_abbrs:
                if dataset_abbr in parsed_results[model_abbr]:
                    row.append(
                        f'{parsed_results[model_abbr][dataset_abbr][metric]:.04f}'  # noqa
                    )
                else:
                    row.append('-')
            table.append(row)
        return table

    def _format_table_pivot(self, table: List[List], decimals: int = 4):
        """Format table as a pandas dataframe and pivot so that columns are
        datasets and rows are models.

        Args:
            table (List[List]): List of lists containing summary table rows
                (including headers)

        Returns:
            pd.DataFrame: Summary dataframe sorted by ascending average BPC
        """
        headers = table.pop(0)
        table_df = pd.DataFrame(table, columns=headers)\
            .drop(columns=['mode'])

        dataset_names = {
            'llm_compression-commoncraw': 'commoncraw',
            'llm_compression-python': 'python',
            'llm_compression-arxiv_math': 'arxiv_math',
        }

        # Pivot model columns to rows
        table_df_long = table_df.melt(id_vars=['dataset', 'version', 'metric'],
                                      var_name='model')

        # Pivot dataset rows to columns
        table_df_wide = table_df_long\
            .pivot(index=['metric', 'version', 'model'], columns='dataset')\
            .droplevel(0, axis=1)\
            .reset_index()\
            .rename(columns=dataset_names)
        table_df_wide.columns.name = None

        # Calculate average BPC per model
        table_df_wide['average'] = table_df_wide[dataset_names.values()]\
            .apply(pd.to_numeric)\
            .mean(axis=1)\
            .round(decimals)

        table_df_wide = table_df_wide[[
            'metric', 'version', 'model', *dataset_names.values(), 'average'
        ]]

        return table_df_wide.sort_values(by='average')\
            .reset_index(drop=True)

    def _output_df_to_file(self, output_path: str, timestamp: str,
                           table: pd.DataFrame) -> None:
        """Output summary dataframe to file.

        Args:
            output_path (str): Output path
            timestamp (str): Timestamp for file suffix
            table (pd.DataFrame): Input dataframe
        """
        if output_path is None:
            output_path = osp.join(self.work_dir, 'summary')

        output_csv_path = osp.join(output_path,
                                   f'summary_pivot_{timestamp}.csv')

        output_dir = osp.split(output_path)[0]
        mmengine.mkdir_or_exist(output_dir)

        table.to_csv(output_csv_path, encoding='utf-8', index=False)
        self.logger.info(f'write csv to {osp.abspath(output_csv_path)}')

    def summarize(
        self,
        output_path: str = None,
        time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):  # noqa
        """Summarize evaluation results and format output table.

        Args:
            output_path (str, optional): Output path. Defaults to None.
            time_str (str, optional): Timestamp for file suffix. Defaults to
            datetime.now().strftime('%Y%m%d_%H%M%S').
        """
        # pick up results
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            self._pick_up_results()

        # calculate group metrics
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            self._calculate_group_metrics(
                raw_results,
                parsed_results,
                dataset_metrics,
                dataset_eval_mode)

        # format table
        table = self._format_table(parsed_results, dataset_metrics,
                                   dataset_eval_mode)

        # convert to list of lists to pandas dataframe and pivot
        table_df = self._format_table_pivot(table)
        with pd.option_context('display.max_columns', 10):
            print(table_df)

        # format raw txt
        raw_txts = self._format_raw_txt(raw_results)

        # output to .text / .csv files
        self._output_to_file(output_path, time_str, table, raw_txts)
        self._output_df_to_file(output_path, time_str, table_df)

        if self.lark_reporter:
            content = f'Detailed evaluation summary for {getpass.getuser()}'
            content += f' saved to {osp.abspath(output_path)}'
            self.lark_reporter.post(content)
