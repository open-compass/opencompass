# flake8: noqa
# yapf: disable
import os.path as osp
from collections import defaultdict
from typing import List, Optional

import mmengine
import numpy as np
from mmengine import ConfigDict
from rich import print
from rich.table import Table

from opencompass.utils import (dataset_abbr_from_cfg, get_infer_output_path,
                               get_logger, model_abbr_from_cfg)
from opencompass.utils.prompt import get_prompt_hash

METRIC_WHITELIST = ['score', 'auc_score', 'accuracy', 'humaneval_pass@1', 'rouge1', 'avg_toxicity_score', 'bleurt_diff', 'matthews_correlation', 'truth']
METRIC_BLACKLIST = ['bp', 'sys_len', 'ref_len']


META_COL_COUNT = 4
EPS = 1e-6

def bold(text):
    return f'[bold]{text}[/bold]'


def green_bold(text):
    return f'[green][bold]{text}[/bold][/green]'


def format_float(v):
    return f'{v:.2f}'


def to_float(text: str):
    try:
        return float(text)
    except ValueError:
        return 0


def is_section_row(row: List[str]) -> bool:
    # ['ceval', '-', '-', '-', '-'],
    return row[-1] == '-' and row[0][0] == '-'


def average_rows(name, rows: List[List[str]]) -> List[str]:
    # name: col=0 的名字
    new_row = ['-'] * len(rows[0])
    new_row[0] = bold(name)

    all_accs = defaultdict(list)
    for row in rows:
        for i, acc in enumerate(row[META_COL_COUNT:]):
            all_accs[i].append(to_float(acc))

    for i, accs in enumerate(all_accs.values()):
        new_row[META_COL_COUNT + i] = format_float(np.mean(accs))
    return new_row


def create_section_row(row_i: int, row: List[str], table) -> List[str]:
    section_name = bold('[' + row[0].replace('-', '').strip() + ']')

    # TODO: 区分 acc,rouge1,score 等
    section_rows = []
    for next_row in table[row_i + 1 :]:
        if is_section_row(next_row):
            break
        section_rows.append(next_row)
    return average_rows(section_name, section_rows)


def create_win_row(rows: List[List[str]]) -> List[str]:
    win_count = defaultdict(int)
    for row in rows:
        all_scores = [to_float(_) for _ in row[META_COL_COUNT:]]
        best_indeice = [i for i, s in enumerate(all_scores) if s > np.max(all_scores) - EPS]
        for best_index in best_indeice:
            win_count[best_index] += 1
    new_row = ['-'] * len(rows[0])
    new_row[0] = bold('Win Count')
    for i, count in win_count.items():
        new_row[META_COL_COUNT + i] = str(count)
    return new_row


def highlight(row: List[str], meta_col_count: int = META_COL_COUNT) -> List[str]:
    new_row = [_ for _ in row]
    all_scores = [to_float(_) for _ in row[meta_col_count:]]
    best_indeice = [i + meta_col_count for i, s in enumerate(all_scores) if s > np.max(all_scores) - EPS]
    for best_index in best_indeice:
        new_row[best_index] = green_bold(row[best_index])
    return new_row


class MultiModelSummarizer:
    """MultiModel.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
        dataset_abbrs (list[str], optional): Dataset abbreviations to be
            listed in the summary.
        summary_groups (list): The dataset groups whose results need to be
            averaged out. For example, mmlu. Each item it a dict with
            'name' (str) and 'subsets' (list of dataset abbrs), and optionally
            'weights' if weighted average is needed.
        prompt_db: A deprecated field.
    """

    def __init__(self, config: ConfigDict, dataset_abbrs: Optional[List[str]] = None, summary_groups: List = [], prompt_db = None) -> None:
        self.tasks = []
        self.cfg = config
        self.logger = get_logger()
        self.summary_groups = summary_groups
        self.dataset_abbrs = dataset_abbrs
        if prompt_db:
            self.logger.warning('prompt_db is deprecated and no longer used. '
                                'Please remove it from your config.')
        self.models_summary_group_metrics = {}
        self.table = self.load()

    def load( self ):  # noqa
        model_cfgs = self.cfg['models']
        dataset_cfgs = self.cfg['datasets']
        work_dir = self.cfg['work_dir']

        # pick up results
        raw_results = {}
        parsed_results = {}
        dataset_metrics = {}

        model_abbrs = [model_abbr_from_cfg(model) for model in model_cfgs]
        for model in model_cfgs:
            model_abbr = model_abbr_from_cfg(model)
            parsed_results[model_abbr] = {}
            raw_results[model_abbr] = {}
            for dataset in dataset_cfgs:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                filepath = get_infer_output_path(model, dataset, osp.join(work_dir, 'results'))
                if not osp.exists(filepath):
                    continue
                result = mmengine.load(filepath)
                raw_results[model_abbr][dataset_abbr] = result
                if 'error' in result:
                    self.logger.debug(f'error in {model_abbr} {dataset_abbr} {result["error"]}')
                    continue
                else:
                    parsed_results[model_abbr][dataset_abbr] = []
                    dataset_metrics[dataset_abbr] = []
                    for metric, score in result.items():
                        if metric not in METRIC_BLACKLIST and isinstance(score, (int, float)):
                            parsed_results[model_abbr][dataset_abbr].append(score)
                            dataset_metrics[dataset_abbr].append(metric)
                        else:
                            continue
                    if len(parsed_results[model_abbr][dataset_abbr]) == 0:
                        self.logger.warning(f'unknown result format: {result}, continue')
                        del parsed_results[model_abbr][dataset_abbr]
                        del dataset_metrics[dataset_abbr]
                        continue
                    indice = sorted(
                        list(range(len(dataset_metrics[dataset_abbr]))),
                        key=lambda i: (
                            METRIC_WHITELIST.index(dataset_metrics[dataset_abbr][i])
                            if dataset_metrics[dataset_abbr][i] in METRIC_WHITELIST
                            else len(METRIC_WHITELIST)
                        )
                    )
                    parsed_results[model_abbr][dataset_abbr] = [parsed_results[model_abbr][dataset_abbr][i] for i in indice]
                    dataset_metrics[dataset_abbr] = [dataset_metrics[dataset_abbr][i] for i in indice]

        # parse eval mode
        dataset_eval_mode = {}
        for dataset in dataset_cfgs:
            inferencer = dataset.get('infer_cfg', {}).get('inferencer', {}).get('type', '')
            inferencer = inferencer if isinstance(inferencer, str) else inferencer.__name__
            dataset_abbr = dataset_abbr_from_cfg(dataset)
            if 'GenInferencer' in inferencer:
                dataset_eval_mode[dataset_abbr] = 'gen'
            elif 'PPLInferencer' in inferencer:
                dataset_eval_mode[dataset_abbr] = 'ppl'
            else:
                dataset_eval_mode[dataset_abbr] = 'unknown'
                self.logger.warning(f'unknown inferencer: {inferencer} - {dataset_abbr}')

        # calculate group metrics
        summary_groups = self.summary_groups
        summary_group_metrics = {}
        for sg in summary_groups:
            for model_abbr in model_abbrs:
                results = {}
                eval_modes = []
                for dataset_abbr in sg['subsets']:
                    if dataset_abbr in parsed_results[model_abbr]:
                        results[dataset_abbr] = parsed_results[model_abbr][dataset_abbr][0]
                        eval_modes.append(dataset_eval_mode.get(dataset_abbr, 'unknown'))
                summary_group_metrics[sg['name']] = results
                if len(results) == len(sg['subsets']):
                    if 'weights' in sg:
                        numerator = sum(results[k] * sg['weights'][k] for k in sg['weights'])
                        denominator = sum(sg['weights'].values())
                        metric = 'weighted_average'
                    else:
                        numerator = sum(results[k] for k in results)
                        denominator = len(results)
                        metric = 'naive_average'
                    results[metric] = numerator / denominator
                    eval_modes = list(set(eval_modes))
                    eval_mode = eval_modes[0] if len(eval_modes) == 1 else 'mixed'

                    # add to global results
                    raw_results[model_abbr][sg['name']] = results
                    parsed_results[model_abbr][sg['name']] = [numerator / denominator]
                    dataset_metrics[sg['name']] = [metric]
                    dataset_eval_mode[sg['name']] = eval_mode
                elif len(results) == 0:
                    continue
                else:
                    raw_results[model_abbr][sg['name']] = {'error': 'missing datasets: {}'.format(set(sg['subsets']) - set(results.keys()))}

        prompt_version = {dataset_abbr_from_cfg(d): get_prompt_hash(d)[:6] for d in dataset_cfgs}

        # format table
        summarizer_dataset_abbrs = []
        if self.dataset_abbrs is None:
            for dataset in dataset_cfgs:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                if dataset_abbr in dataset_metrics:
                    for metric in dataset_metrics[dataset_abbr]:
                        summarizer_dataset_abbrs.append((dataset_abbr, metric))
                else:
                    summarizer_dataset_abbrs.append((dataset_abbr, None))
            for dataset_abbr in dataset_metrics:
                for metric in dataset_metrics[dataset_abbr]:
                    if (dataset_abbr, metric) not in summarizer_dataset_abbrs:
                        summarizer_dataset_abbrs.append((dataset_abbr, metric))
        else:
            for item in self.dataset_abbrs:
                if isinstance(item, str):
                    summarizer_dataset_abbrs.append((item, None))
                elif isinstance(item, (list, tuple)):
                    summarizer_dataset_abbrs.append((item[0], item[1]))

        table = []
        header = ['dataset', 'version', 'metric', 'mode'] + model_abbrs
        table.append(header)
        for dataset_abbr, metric in summarizer_dataset_abbrs:
            if dataset_abbr not in dataset_metrics:
                table.append([dataset_abbr, '-', '-', '-'] + ['-'] * len(model_abbrs))
                continue
            if metric is None:
                index = 0
                metric = dataset_metrics[dataset_abbr][0]
            elif metric in dataset_metrics[dataset_abbr]:
                index = dataset_metrics[dataset_abbr].index(metric)
            else:
                table.append([dataset_abbr, '-', '-', '-'] + ['-'] * len(model_abbrs))
                continue

            row = [dataset_abbr, prompt_version.get(dataset_abbr, '-'), metric, dataset_eval_mode.get(dataset_abbr, '-')]
            for model_abbr in model_abbrs:
                if dataset_abbr in parsed_results[model_abbr]:
                    row.append('{:.02f}'.format(parsed_results[model_abbr][dataset_abbr][index]))
                else:
                    row.append('-')
            table.append(row)

        self.models_summary_group_metrics[table[0][-1]] = summary_group_metrics
        return table

    def merge(self, summarizer: 'MultiModelSummarizer'):
        assert len(self.table) == len(summarizer.table)
        for row_i, row in enumerate(summarizer.table):
            base_row = self.table[row_i]
            if base_row[:3] != row[:3]:
                self.logger.warning(f'cannot merge tables with different headers: {base_row} vs {row}')
            base_row.extend(row[META_COL_COUNT:])
        new_model_name = summarizer.table[0][-1]
        assert new_model_name not in self.models_summary_group_metrics
        self.models_summary_group_metrics[new_model_name] = summarizer.models_summary_group_metrics[new_model_name]

    def summarize(self):
        """
        Format in self.table
        [
            ['dataset', 'version', 'metric', 'mode', 'model_name'],
            ['--------- 考试 Exam ---------', '-', '-', '-', '-'],
            ['ARC-c', '1e0de5', 'accuracy', 'gen', '79.32'],
            ['ARC-e', '1e0de5', 'accuracy', 'gen', '85.36'],
            ['--------- 语言 Language ---------', '-', '-', '-', '-'],
            ['WiC', 'd06864', 'accuracy', 'gen', '55.64'],
            ['chid-dev', '211ee7', 'accuracy', 'gen', '52.97'],
            ['--------- 知识 Knowledge ---------', '-', '-', '-', '-'],
            ['BoolQ', '883d50', 'accuracy', 'gen', '86.06'],
            ['--------- 理解 Understanding ---------', '-', '-', '-', '-'],
            ['C3', '8c358f', 'accuracy', 'gen', '88.33'],
            ['race-middle', '9a54b6', 'accuracy', 'gen', '90.32'],
            ['--------- 推理 Reasoning ---------', '-', '-', '-', '-'],
            ['cmnli', '1abf97', 'accuracy', 'gen', '38.26'],
            ['ocnli', 'c4cb6c', 'accuracy', 'gen', '32.92'],
        ]
        """

        table = Table()
        for i, col_name in enumerate(self.table[0]):
            table.add_column(col_name, overflow='fold', max_width=20 if i >= META_COL_COUNT else None)

        section_rows = []
        all_rows = []
        for row_i, row in enumerate(self.table):
            if row_i == 0:
                continue
            if is_section_row(row):
                table.add_section()
                new_row = create_section_row(row_i, row, self.table)
                section_rows.append(new_row)
            else:
                new_row = row
                all_rows.append(new_row)

            table.add_row(*highlight(new_row))

        if section_rows:
            table.add_section()
            average_row = average_rows('Naive Average', section_rows)
            average_row = highlight(average_row)
            table.add_row(*average_row)

        table.add_section()
        table.add_row(*highlight(create_win_row(all_rows)))
        print(table)

    def show_group(self, group: str):
        table = Table(title=group)
        table.add_column('Dataset', overflow='fold')

        # summary_group_metrics 数据结构 dict[group_name][sub_group_name] = 73
        group_metrics = None
        for model_name, summary_group_metrics in self.models_summary_group_metrics.items():
            if group not in summary_group_metrics:
                self.logger.warning(f'group {group} not found in {model_name}')
                return
            table.add_column(model_name, overflow='fold')
            group_metrics = summary_group_metrics[group]

        for subset_name in group_metrics.keys():
            if subset_name == 'naive_average':
                continue

            row = [subset_name]
            for summary_group_metrics in self.models_summary_group_metrics.values():
                metric = summary_group_metrics[group][subset_name]
                row.append(format_float(metric))
            table.add_row(*highlight(row, meta_col_count=1))

        print(table)
