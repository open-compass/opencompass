# flake8: noqa
# yapf: disable
import functools
import getpass
import math
import os.path as osp
from datetime import datetime
from typing import Any, Dict, List, Optional

import mmengine
import tabulate
from mmengine import ConfigDict

from opencompass.utils import (LarkReporter, dataset_abbr_from_cfg,
                               get_infer_output_path, get_logger,
                               model_abbr_from_cfg)
from opencompass.utils.prompt import get_prompt_hash

METRIC_WHITELIST = ['score', 'auc_score', 'accuracy', 'humaneval_pass@1', 'rouge1', 'avg_toxicity_score', 'bleurt_diff', 'matthews_correlation', 'truth', 'f1', 'exact_match', 'extract_rate']
METRIC_BLACKLIST = ['bp', 'sys_len', 'ref_len', 'type']

def model_abbr_from_cfg_used_in_summarizer(model):
    if model.get('summarizer_abbr', None):
        return model['summarizer_abbr']
    else:
        return model_abbr_from_cfg(model)


class DefaultSummarizer:
    """Default summarizer in OpenCompass.

    Args:
        config (ConfigDict): The configuration object of the evaluation task. It's expected to be filled out at runtime.
        dataset_abbrs (list[str], optional): Dataset abbreviations to be listed in the summary.
        summary_groups (list): The dataset groups whose results need to be averaged out. For example, mmlu. Each item it a dict with
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

        # Enable lark bot if lark_url is presented
        self.lark_reporter = None
        if self.cfg.get('lark_bot_url', None):
            self.lark_reporter = LarkReporter(self.cfg['lark_bot_url'])

        self.model_cfgs = self.cfg['models']
        self.dataset_cfgs = self.cfg['datasets']
        self.work_dir = self.cfg['work_dir']
        model_abbrs = []
        for model in self.model_cfgs:
            model_abbr = model_abbr_from_cfg_used_in_summarizer(model)
            if model_abbr in model_abbrs:
                continue
            model_abbrs.append(model_abbr)
        self.model_abbrs = model_abbrs

    def _pick_up_results(self):
        """The function reads the numerical results of evaluations from the
        output folder based on the configuration file, and ultimately returns
        four dictionaries, each containing processed information in different
        formats. The contents of the four dictionaries are as follows:

        - raw_results: contains the raw results of each model on each dataset (excluding details).
        - parsed_results: contains the results of each model on each dataset for each metric, with metrics in METRIC_BLACKLIST being ignored.
        - dataset_metrics: contains the list of metrics for each dataset, consistent with the metrics in parsed_results. The list is ordered according to the METRIC_WHITELIST,
            with metrics appearing earlier considered more important.
        - dataset_eval_mode: contains the evaluation mode for each dataset.
        """
        # raw_results: {model_abbr: {dataset_abbr: result}}
        raw_results : Dict[str, Dict[str, Any]] = {}
        # parsed_results: {model_abbr: {dataset_abbr: {metric: score}}}
        parsed_results : Dict[str, Dict[str, Dict[str, float]]] = {}
        # dataset_metrics: {dataset_abbr: [metric]}
        dataset_metrics : Dict[str, List[str]] = {}

        for model in self.model_cfgs:
            model_abbr = model_abbr_from_cfg_used_in_summarizer(model)
            parsed_results.setdefault(model_abbr, {})
            raw_results.setdefault(model_abbr, {})
            for dataset in self.dataset_cfgs:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                filepath = get_infer_output_path(model, dataset, osp.join(self.work_dir, 'results'))
                if not osp.exists(filepath):
                    continue
                result = mmengine.load(filepath)
                result.pop('details', None)
                raw_results[model_abbr][dataset_abbr] = result
                if 'error' in result:
                    self.logger.debug(f'error in {model_abbr} {dataset_abbr} {result["error"]}')
                    continue
                _rst, _dm = {}, []
                for metric, score in result.items():
                    if metric not in METRIC_BLACKLIST and isinstance(score, (int, float)):
                        _rst[metric] = score
                        _dm.append(metric)
                    else:
                        continue
                if len(_rst) == 0:
                    self.logger.warning(f'unknown result format: {result}, continue')
                    continue
                _dm = sorted(_dm, key=lambda i: METRIC_WHITELIST.index(i) if i in METRIC_WHITELIST else len(METRIC_WHITELIST))

                if dataset_abbr in dataset_metrics:
                    assert tuple(dataset_metrics[dataset_abbr]) == tuple(_dm), \
                    f'{dataset_abbr} has different metrics: {dataset_metrics[dataset_abbr]} vs {_dm}'
                else:
                    dataset_metrics[dataset_abbr] = _dm
                parsed_results[model_abbr][dataset_abbr] = _rst

        # dataset_eval_mode: {dataset_abbr: eval_mode}
        dataset_eval_mode : Dict[str, str] = {}
        for dataset in self.dataset_cfgs:
            inferencer = dataset.get('infer_cfg', {}).get('inferencer', {}).get('type', '')
            inferencer = inferencer if isinstance(inferencer, str) else inferencer.__name__
            dataset_abbr = dataset_abbr_from_cfg(dataset)
            if 'GenInferencer' in inferencer:
                dataset_eval_mode[dataset_abbr] = 'gen'
            elif 'PPLInferencer' in inferencer:
                dataset_eval_mode[dataset_abbr] = 'ppl'
            elif 'LLInferencer' in inferencer:
                dataset_eval_mode[dataset_abbr] = 'll'
            else:
                dataset_eval_mode[dataset_abbr] = 'unknown'
                self.logger.warning(f'unknown inferencer: {inferencer} - {dataset_abbr}')
        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode

    def _calculate_group_metrics(self, raw_results, parsed_results, dataset_metrics, dataset_eval_mode):
        """The function calculates the numerical results for each group based
        on the configuration in summary_groups, and updates the contents of
        each dictionary accordingly."""
        summary_groups = self.summary_groups
        for sg in summary_groups:
            for model_abbr in self.model_abbrs:
                available_metrics, missing_metrics = [], []
                for i in sg['subsets']:
                    if isinstance(i, (list, tuple)):
                        if i[0] in parsed_results[model_abbr] and i[1] in parsed_results[model_abbr][i[0]]:
                            available_metrics.append(i)
                        else:
                            missing_metrics.append(i)
                    else:
                        if i in parsed_results[model_abbr]:
                            available_metrics.append(i)
                        else:
                            missing_metrics.append(i)

                if len(available_metrics) == 0:
                    continue
                if len(missing_metrics) != 0:
                    raw_results[model_abbr][sg['name']] = {'error': 'missing metrics: {}'.format(missing_metrics)}
                    continue

                if 'metric' in sg:
                    default_metric = sg['metric']
                    need_smart_metric = False
                else:
                    need_smart_metric = True
                    if sg.get('std', False):
                        default_metric = 'standard_deviation'
                    elif sg.get('sum', False):
                        default_metric = 'sum'
                    elif sg.get('weights', []):
                        default_metric = 'weighted_average'
                    else:
                        default_metric = 'naive_average'

                scores, eval_modes, group_metrics = {}, [], None
                if any(isinstance(dataset_abbr, (list, tuple)) for dataset_abbr in sg['subsets']) and \
                    any(isinstance(dataset_abbr, str) for dataset_abbr in sg['subsets']):
                    raise NotImplementedError('mixed dataset_abbr type is not supported')

                if all(isinstance(dataset_abbr, (list, tuple)) for dataset_abbr in sg['subsets']):
                    group_metrics = [default_metric]
                    for dataset_abbr, metric in sg['subsets']:
                        scores.setdefault(default_metric, {})[dataset_abbr + '@' + metric] = parsed_results[model_abbr][dataset_abbr][metric]
                        eval_modes.append(dataset_eval_mode.get(dataset_abbr, 'unknown'))
                else:
                    group_metrics = list(functools.reduce(lambda a, b: a & b, [set(dataset_metrics[dataset_abbr]) for dataset_abbr in sg['subsets']]))
                    if need_smart_metric and len(group_metrics) > 1:
                        for metric in group_metrics:
                            for dataset_abbr in sg['subsets']:
                                scores.setdefault(metric, {})[dataset_abbr + '@' + metric] = parsed_results[model_abbr][dataset_abbr][metric]
                                eval_modes.append(dataset_eval_mode.get(sg['subsets'][0], 'unknown'))
                    else:
                        group_metrics = [default_metric]
                        for dataset_abbr in sg['subsets']:
                            metric = dataset_metrics[dataset_abbr][0]
                            scores.setdefault(default_metric, {})[dataset_abbr + '@' + metric] = parsed_results[model_abbr][dataset_abbr][metric]
                            eval_modes.append(dataset_eval_mode.get(dataset_abbr, 'unknown'))

                result = {}
                for metric in scores:
                    if default_metric == 'standard_deviation':
                        avg = sum(scores[metric].values()) / len(scores[metric])
                        variance = sum((scores[metric][k] - avg) ** 2 for k in scores[metric]) / len(scores[metric])
                        scores[metric] = result[metric] = math.sqrt(variance)
                    else:
                        if sg.get('weights', []):
                            # check sg['weights'][k] != 0 in case of scores[metric][k] is NaN
                            try:
                                numerator = sum(scores[metric][k] * sg['weights'][k] for k in sg['weights'] if sg['weights'][k] != 0)
                            except KeyError:
                                tmp_scores = {metric: {k.split('@')[0]: v for k, v in scores[metric].items()}}
                                numerator = sum(tmp_scores[metric][k] * sg['weights'][k] for k in sg['weights'] if sg['weights'][k] != 0)
                            denominator = sum(sg['weights'].values())
                        else:
                            numerator = sum(scores[metric].values())
                            denominator = len(scores[metric])
                        if default_metric == 'sum':
                            scores[metric] = result[metric] = numerator
                        else:
                            scores[metric] = result[metric] = numerator / denominator
                    eval_modes = list(set(eval_modes))
                    eval_mode = eval_modes[0] if len(eval_modes) == 1 else 'mixed'

                # add to global results
                raw_results[model_abbr].setdefault(sg['name'], {}).update(scores)
                parsed_results[model_abbr].setdefault(sg['name'], {}).update(result)
                dataset_metrics.setdefault(sg['name'], []).extend(group_metrics)
                dataset_eval_mode[sg['name']] = eval_mode

        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode

    def _format_table(self, parsed_results, dataset_metrics, dataset_eval_mode, required_dataset_abbrs=None, skip_all_slash=False):
        dataset_abbrs = [dataset_abbr_from_cfg(dataset) for dataset in self.dataset_cfgs]
        prompt_version = {dataset_abbr_from_cfg(d): get_prompt_hash(d)[:6] for d in self.dataset_cfgs}

        summarizer_dataset_abbrs = []
        if required_dataset_abbrs is None:
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
            for item in required_dataset_abbrs:
                if isinstance(item, str):
                    summarizer_dataset_abbrs.append((item, None))
                elif isinstance(item, (list, tuple)):
                    summarizer_dataset_abbrs.append((item[0], item[1]))

        table = []
        header = ['dataset', 'version', 'metric', 'mode'] + self.model_abbrs
        table.append(header)
        for dataset_abbr, metric in summarizer_dataset_abbrs:
            if dataset_abbr not in dataset_metrics:
                if not skip_all_slash:
                    table.append([dataset_abbr, '-', '-', '-'] + ['-'] * len(self.model_abbrs))
                continue
            if metric is None:
                metric = dataset_metrics[dataset_abbr][0]
            elif metric in dataset_metrics[dataset_abbr]:
                pass
            else:
                if not skip_all_slash:
                    table.append([dataset_abbr, '-', '-', '-'] + ['-'] * len(self.model_abbrs))
                continue

            row = [dataset_abbr, prompt_version.get(dataset_abbr, '-'), metric, dataset_eval_mode.get(dataset_abbr, '-')]
            for model_abbr in self.model_abbrs:
                if dataset_abbr in parsed_results[model_abbr]:
                    row.append('{:.02f}'.format(parsed_results[model_abbr][dataset_abbr][metric]))
                else:
                    row.append('-')
            table.append(row)
        return table

    def _format_raw_txt(self, raw_results):
        raw_dataset_abbrs = []
        for model_abbr in self.model_abbrs:
            for dataset_abbr in raw_results[model_abbr]:
                if dataset_abbr not in raw_dataset_abbrs:
                    raw_dataset_abbrs.append(dataset_abbr)
        raw_txts = []
        for model_abbr in self.model_abbrs:
            raw_txts.append('-------------------------------')
            raw_txts.append(f'Model: {model_abbr}')
            for dataset_abbr in raw_dataset_abbrs:
                result = raw_results[model_abbr].get(dataset_abbr, '{}')
                raw_txts.append(f'{dataset_abbr}: {result}')
        raw_txts = '\n'.join(raw_txts)
        return raw_txts

    def _output_to_file(self, output_path, time_str, table, raw_txts):
        # output to file
        if output_path is None:
            output_path = osp.join(self.work_dir, 'summary', f'summary_{time_str}.txt')
            output_csv_path = osp.join(self.work_dir, 'summary', f'summary_{time_str}.csv')
        else:
            output_csv_path = output_path.replace('.txt', '.csv')

        output_dir = osp.split(output_path)[0]
        mmengine.mkdir_or_exist(output_dir)
        with open(output_path, 'w', encoding='utf-8') as f:
            text = f'{time_str}\n' + \
                    'tabulate format\n' + \
                    '^' * 128 + '\n' + \
                    tabulate.tabulate(table, headers='firstrow', floatfmt='.2f') + '\n' + \
                    '$' * 128 + '\n\n' + \
                    '-' * 128 + ' THIS IS A DIVIDER ' + '-' * 128 + '\n\n' + \
                    'csv format\n' + \
                    '^' * 128 + '\n' + \
                    '\n'.join([','.join(row) for row in table]) + '\n' + \
                    '$' * 128 + '\n\n' + \
                    '-' * 128 + ' THIS IS A DIVIDER ' + '-' * 128 + '\n\n' + \
                    'raw format\n' + \
                    '^' * 128 + '\n' + \
                    raw_txts + '\n' + \
                    '$' * 128 + '\n'
            f.write(text)
        self.logger.info(f'write summary to {osp.abspath(output_path)}')

        with open(output_csv_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join([','.join(row) for row in table]) + '\n')
        self.logger.info(f'write csv to {osp.abspath(output_csv_path)}')

    def summarize(
        self,
        output_path: str = None,
        time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):  # noqa

        # pick up results
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = self._pick_up_results()

        # calculate group metrics
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            self._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # format table
        table = self._format_table(parsed_results, dataset_metrics, dataset_eval_mode, required_dataset_abbrs=self.dataset_abbrs)

        # format raw txt
        raw_txts = self._format_raw_txt(raw_results)

        # output to screen
        print(tabulate.tabulate(table, headers='firstrow', floatfmt='.2f'))

        # output to .text / .csv files
        self._output_to_file(output_path, time_str, table, raw_txts)

        if self.lark_reporter:
            content = f'{getpass.getuser()} 的'
            content += f'详细评测汇总已输出至 {osp.abspath(output_path)}'
            self.lark_reporter.post(content)
