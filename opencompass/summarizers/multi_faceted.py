# flake8: noqa
# yapf: disable
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import tabulate
from mmengine import ConfigDict

from .default import DefaultSummarizer


class MultiFacetedSummarizer(DefaultSummarizer):

    def __init__(self, config: ConfigDict, dataset_abbrs_list: Optional[Dict[str, List[str]]] = None, summary_groups: List = []) -> None:
        super().__init__(config, dataset_abbrs=None, summary_groups=summary_groups)
        self.dataset_abbrs_list = dataset_abbrs_list

    def summarize(self, output_path: str = None, time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):

        # pick up results
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = self._pick_up_results()

        # calculate group metrics
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            self._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        for dataset_abbrs_item in self.dataset_abbrs_list:
            profile_name = dataset_abbrs_item['name']
            profile_dataset_abbrs = dataset_abbrs_item['dataset_abbrs']

            # format table
            table = self._format_table(parsed_results, dataset_metrics, dataset_eval_mode, required_dataset_abbrs=profile_dataset_abbrs, skip_all_slash=True)
            if len(table) == 1:
                continue

            # output to screen
            print(tabulate.tabulate(table, headers='firstrow', floatfmt='.2f'))

            # output to .text / .csv files
            output_csv_path = os.path.join(self.work_dir, 'summary', f'summary_{time_str}', f'{profile_name}.csv')
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join([','.join(row) for row in table]) + '\n')
            self.logger.info(f'write csv to {os.path.abspath(output_csv_path)}')
