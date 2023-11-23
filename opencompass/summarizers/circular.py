from typing import List, Optional

from mmengine import ConfigDict

from opencompass.utils import dataset_abbr_from_cfg
from opencompass.utils.prompt import get_prompt_hash

from .default import DefaultSummarizer


class CircularSummarizer(DefaultSummarizer):

    def __init__(self,
                 config: ConfigDict,
                 dataset_abbrs: Optional[List[str]] = None,
                 summary_groups: List = [],
                 prompt_db=None,
                 metric_types=None) -> None:
        super().__init__(config, dataset_abbrs, summary_groups, prompt_db)
        self.metric_types = metric_types

    def _format_table(self, parsed_results, dataset_metrics,
                      dataset_eval_mode):
        prompt_version = {
            dataset_abbr_from_cfg(d): get_prompt_hash(d)[:6]
            for d in self.dataset_cfgs
        }

        table = []
        header1 = ['dataset', 'version', 'mode'] + sum(
            [[model_abbr] + ['-' for _ in range(len(self.metric_types) - 1)]
             for model_abbr in self.model_abbrs], [])
        table.append(header1)
        header2 = ['-', '-', '-'] + sum(
            [self.metric_types for _ in self.model_abbrs], [])
        table.append(header2)
        for dataset_abbr in self.dataset_abbrs:
            if dataset_abbr not in dataset_metrics:
                table.append([dataset_abbr, '-', '-'] + ['-'] *
                             len(self.model_abbrs) * len(self.metric_types))
                continue
            row = [
                dataset_abbr,
                prompt_version.get(dataset_abbr, '-'),
                dataset_eval_mode.get(dataset_abbr, '-')
            ]
            for model_abbr in self.model_abbrs:
                for metric in self.metric_types:
                    if dataset_abbr in parsed_results[
                            model_abbr] and metric in parsed_results[
                                model_abbr][dataset_abbr]:
                        row.append('{:.02f}'.format(
                            parsed_results[model_abbr][dataset_abbr][metric]))
                    else:
                        row.append('-')
            table.append(row)
        return table
