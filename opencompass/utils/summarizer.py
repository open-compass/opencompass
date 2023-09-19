# flake8: noqa
# yapf: disable
import getpass
import os.path as osp
from datetime import datetime

import mmengine
import tabulate
from mmengine import ConfigDict

from opencompass.utils import (LarkReporter, dataset_abbr_from_cfg,
                               get_infer_output_path, get_logger,
                               model_abbr_from_cfg)
from opencompass.utils.prompt import get_prompt_hash

METRIC_WHITELIST = ['score', 'auc_score', 'accuracy', 'humaneval_pass@1', 'rouge1', 'avg_toxicity_score', 'bleurt_diff', 'matthews_correlation', 'truth']
METRIC_BLACKLIST = ['bp', 'sys_len', 'ref_len']

class Summarizer:
    """"""

    def __init__(self, config: ConfigDict) -> None:
        self.tasks = []
        self.cfg = config
        self.logger = get_logger()

        # Enable lark bot if lark_url is presented
        self.lark_reporter = None
        if self.cfg.get('lark_bot_url', None):
            self.lark_reporter = LarkReporter(self.cfg['lark_bot_url'])

    def summarize(
        self,
        output_path: str = None,
        time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):  # noqa

        model_cfgs = self.cfg['models']
        dataset_cfgs = self.cfg['datasets']
        summarizer_cfg = self.cfg.get('summarizer', {}) or {} # avoid 'summarizer' is in cfg but None
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
                    self.debug(f'error in {model_abbr} {dataset_abbr} {result["error"]}')
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
        summary_groups = summarizer_cfg.get('summary_groups', [])
        for sg in summary_groups:
            for model_abbr in model_abbrs:
                results = {}
                eval_modes = []
                for dataset_abbr in sg['subsets']:
                    if dataset_abbr in parsed_results[model_abbr]:
                        results[dataset_abbr] = parsed_results[model_abbr][dataset_abbr][0]
                        eval_modes.append(dataset_eval_mode.get(dataset_abbr, 'unknown'))
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
        if summarizer_cfg.get('dataset_abbrs') is None:
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
            for item in summarizer_cfg['dataset_abbrs']:
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

        # format raw txt
        raw_dataset_abbrs = []
        for model_abbr in model_abbrs:
            for dataset_abbr in raw_results[model_abbr]:
                if dataset_abbr not in raw_dataset_abbrs:
                    raw_dataset_abbrs.append(dataset_abbr)
        raw_txts = []
        for model_abbr in model_abbrs:
            raw_txts.append('-------------------------------')
            raw_txts.append(f'Model: {model_abbr}')
            for dataset_abbr in raw_dataset_abbrs:
                result = raw_results[model_abbr].get(dataset_abbr, '{}')
                raw_txts.append(f'{dataset_abbr}: {result}')
        raw_txts = '\n'.join(raw_txts)

        # output to screean
        print(tabulate.tabulate(table, headers='firstrow'))

        # output to file
        if output_path is None:
            output_path = osp.join(work_dir, 'summary', f'summary_{time_str}.txt')
            output_csv_path = osp.join(work_dir, 'summary', f'summary_{time_str}.csv')
        else:
            output_csv_path = output_path.replace('.txt', '.csv')

        output_dir = osp.split(output_path)[0]
        mmengine.mkdir_or_exist(output_dir)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(time_str + '\n')
            f.write('tabulate format\n')
            f.write('^' * 128 + '\n')
            f.write(tabulate.tabulate(table, headers='firstrow') + '\n')
            f.write('$' * 128 + '\n')
            f.write('\n' + '-' * 128 + ' THIS IS A DIVIDER ' + '-' * 128 + '\n\n')
            f.write('csv format\n')
            f.write('^' * 128 + '\n')
            f.write('\n'.join([','.join(row) for row in table]) + '\n')
            f.write('$' * 128 + '\n')
            f.write('\n' + '-' * 128 + ' THIS IS A DIVIDER ' + '-' * 128 + '\n\n')
            f.write('raw format\n')
            f.write('^' * 128 + '\n')
            f.write(raw_txts + '\n')
            f.write('$' * 128 + '\n')
        self.logger.info(f'write summary to {osp.abspath(output_path)}')

        if self.lark_reporter:
            content = f'{getpass.getuser()} 的'
            content += f'详细评测汇总已输出至 {osp.abspath(output_path)}'
            self.lark_reporter.post(content)

        with open(output_csv_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join([','.join(row) for row in table]) + '\n')
        self.logger.info(f'write csv to {osp.abspath(output_csv_path)}')
