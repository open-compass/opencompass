# flake8: noqa
# yapf: disable
import argparse
import functools
import getpass
import math
import os
import os.path as osp
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mmengine
import numpy as np
import pandas as pd
import seaborn as sns
import tabulate
from matplotlib.colors import LinearSegmentedColormap
from mmengine import ConfigDict
from tqdm import tqdm

from opencompass.utils import (LarkReporter, dataset_abbr_from_cfg,
                               get_infer_output_path, get_logger,
                               model_abbr_from_cfg)
from opencompass.utils.prompt import get_prompt_hash

METRIC_WHITELIST = ['score', 'auc_score', 'accuracy', 'humaneval_pass@1', 'rouge1', 'avg_toxicity_score', 'bleurt_diff', 'matthews_correlation', 'truth']
METRIC_BLACKLIST = ['bp', 'sys_len', 'ref_len']

def model_abbr_from_cfg_used_in_summarizer(model):
    if model.get('summarizer_abbr', None):
        return model['summarizer_abbr']
    else:
        return model_abbr_from_cfg(model)

def read_after_specific_line_except_last(file_name, keyword, offset):
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 找到含有关键词的行的索引
    for index, line in enumerate(lines):
        if keyword in line:
            start_index = index + offset + 1
            break
    else:
        # 如果文件中没有找到关键字，则返回空字符串
        return ''

    # 返回从指定行开始到倒数第二行的内容
    return ''.join(lines[start_index:-1])

def create_model_dataframe(nested_dict, model_name, dataset_abbr, parallel=False):
    # 确保模型名存在于字典中
    if model_name not in nested_dict:
        print(f'Model {model_name} not found in the provided data.')
        return pd.DataFrame()  # 返回一个空的DataFrame

    model_data = nested_dict[model_name]
    data = []

    for key, value in model_data.items():
        if parallel:
            if dataset_abbr in key:
                new_key_base = key.replace(dataset_abbr, '').strip('_')
                for depth_key, score in value.items():
                    new_key = f'{new_key_base}{depth_key}'
                    if 'average_score' not in new_key:
                        data.append([new_key, score])
        else:
            if dataset_abbr in key:
                score = value.get('score', None)
                new_key = key.replace(dataset_abbr, '').strip('_')
                data.append([new_key, score])

    df = pd.DataFrame(data, columns=['dataset', model_name])
    return df

def parse_model_scores(text):
    # 分割字符串为多行
    lines = text.split('\n')

    result_dict = {}
    current_model = None

    for line in lines:
        # 检查行是否定义了新模型
        if line.startswith('Model:'):
            # 获取模型名称
            current_model = line.split('Model:')[1].strip()
            result_dict[current_model] = {}
        elif current_model and ':' in line:
            # 解析数据集和分数
            dataset, score_str = line.split(':', 1)
            score_dict = eval(score_str.strip())
            result_dict[current_model][dataset] = score_dict

    return result_dict

def visualize(df_raw, save_path: str,model_name: str ,dataset_type:str):
    df = df_raw.copy()
    df['Context Length'] = df['dataset'].apply(
        lambda x: int(x.split('Length')[1].split('Depth')[0]))
    df['Document Depth'] = df['dataset'].apply(
        lambda x: float(x.split('Depth')[1].split('_')[0]))

    model_columns = [
        col for col in df.columns
        if col not in ['Context Length', 'Document Depth']
    ]

    for model_name in model_columns[1:]:
        model_df = df[['Document Depth', 'Context Length',
                        model_name]].copy()
        model_df.rename(columns={model_name: 'Score'}, inplace=True)

        # Create pivot table
        pivot_table = pd.pivot_table(model_df,
                                        values='Score',
                                        index=['Document Depth'],
                                        columns=['Context Length'],
                                        aggfunc='mean')

        # Calculate mean scores
        mean_scores = pivot_table.mean().values

        # Calculate overall score
        overall_score = mean_scores.mean()

        # Create heatmap and line plot
        plt.figure(figsize=(15.5, 8))
        ax = plt.gca()
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap', ['#F0496E', '#EBB839', '#0CD79F'])

        # Draw heatmap
        sns.heatmap(pivot_table,
                    cmap=cmap,
                    ax=ax,
                    cbar_kws={'label': 'Score'},
                    vmin=0,
                    vmax=100)

        # Set line plot data
        x_data = [i + 0.5 for i in range(len(mean_scores))]
        y_data = mean_scores

        # Create twin axis for line plot
        ax2 = ax.twinx()
        # Draw line plot
        ax2.plot(x_data,
                    y_data,
                    color='white',
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=8,
                    label='Average Depth Score')
        # Set y-axis range
        ax2.set_ylim(0, 100)

        # Hide original y-axis ticks and labels
        ax2.set_yticklabels([])
        ax2.set_yticks([])

        # Add legend
        ax2.legend(loc='upper left')

        # Set chart title and labels
        ax.set_title(f'{model_name} {dataset_type} Context '
                        'Performance\nFact Retrieval Across '
                        'Context Lengths ("Needle In A Haystack")')
        ax.set_xlabel('Token Limit')
        ax.set_ylabel('Depth Percent')
        ax.set_xticklabels(pivot_table.columns.values, rotation=45)
        ax.set_yticklabels(pivot_table.index.values, rotation=0)
        # Add overall score as a subtitle
        plt.text(0.5,
                 -0.13, f'Overall Score for {model_name}: '
                 f'{overall_score:.2f}',
                 ha='center',
                 va='center',
                 transform=ax.transAxes,
                 fontsize=13)

        plt.tight_layout()
        plt.subplots_adjust(right=1)
        plt.draw()
        plt.savefig(save_path)
        print(f'Saved :{save_path}')
        plt.close()  # Close figure to prevent memory leaks
    return overall_score

def save_results_to_plots(txt_results_save_path):

    content = read_after_specific_line_except_last(txt_results_save_path, 'raw format', 2)

    parsed_data = parse_model_scores(content)
    model_names = get_dict_model_names(parsed_data)
    # 定义数字、语言代码和尺寸
    numbers = [2, 3, 4, 5]
    languages = ['en', 'zh']
    size_exists = []
    sizes_origin = ['_4k', '_8k', '_32k', '_128k', '_200k']

    for size in sizes_origin:
        if size in content:
            size_exists.append(size)

    # 创建dataset_abbrs列表
    multi_dataset_abbrs = [f'{num}needle_{lang}{size}' for num in numbers for lang in languages for size in size_exists]
    origin_dataset_abbrs = [f'origin_{lang}{size}' for lang in languages for size in size_exists]
    # 创建parallel_dataset_abbrs列表
    parallel_dataset_abbrs = [f'parallel_{lang}{size}' for lang in languages for size in size_exists]

    dataset_abbrs = multi_dataset_abbrs + origin_dataset_abbrs + \
                        parallel_dataset_abbrs
    base_path = os.path.dirname(txt_results_save_path)
    plot_path = os.path.join(base_path, 'plots')
    model_scores = {}
    for model_name in tqdm(model_names):
        model_datasets_scores = {}  # Dictionary to store scores for each dataset for the current model
        for dataset_abbr in dataset_abbrs:
            parallel_flag = 'parallel' in dataset_abbr

            # Create a directory for each dataset_abbr
            folder_path = os.path.join(plot_path, dataset_abbr)
            ensure_directory(folder_path)

            # Construct the full path to save the image
            save_path = os.path.join(folder_path, f'{model_name}.png')

            # Create DataFrame for the model and dataset
            df = create_model_dataframe(parsed_data, model_name, dataset_abbr, parallel=parallel_flag)

            # Generate visualization and get the score
            score = visualize(df, save_path, model_name, dataset_abbr)

            # Store the score in the dictionary
            model_datasets_scores[dataset_abbr] = '{:.02f}'.format(score)

        # Process and visualize the overall score
        overall_score_pic_path = os.path.join(plot_path, f'{model_name}_overall.png')
        merged_df = merge_dataframes(model_name, dataset_abbrs, parsed_data)
        averaged_df = calculate_elementwise_average(merged_df)

        # Assume visualize returns the average score for the overall visualization
        overall_score = visualize(averaged_df, overall_score_pic_path, 'weighted_average_score', 'Overall Score')

        # Add the overall score to the dictionary
        model_datasets_scores['Overall'] = '{:.02f}'.format(overall_score)

        # Add the model's scores to the main dictionary
        model_scores[model_name] = model_datasets_scores

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_dict_model_names(nested_dict):
    model_names = []
    for first_level_key in nested_dict:
        model_names.append(first_level_key)
    return model_names

def merge_dataframes(model_name, dataset_abbrs, parsed_data):
    dfs = []
    for dataset_abbr in dataset_abbrs:
        parallel_flag = 'parallel' in dataset_abbr
        df = create_model_dataframe(parsed_data, model_name, dataset_abbr, parallel=parallel_flag)

        # 检查DataFrame是否为空或是否有多于一列（除了'dataset'列）
        if not df.empty and len(df.columns) > 1:
            # 将模型名称列重命名为dataset_abbr
            score_column = df.columns[-1]  # 假设分数列是最后一列
            df.rename(columns={score_column: dataset_abbr}, inplace=True)

        dfs.append(df)

    # 沿着列方向合并DataFrame
    merged_df = pd.concat(dfs, axis=1)
    return merged_df

def calculate_elementwise_average(merged_df):
    # 选择需要计算平均值的列
    score_columns = [col for col in merged_df.columns if col != 'dataset']

    origin_columns = [col for col in score_columns if 'origin' in col]
    parallel_columns = [col for col in score_columns if 'parallel' in col]
    multi_columns = [col for col in score_columns if 'needle'  in col]

    # 计算加权平均分数
    if origin_columns and parallel_columns and multi_columns:
        origin_avg = merged_df[origin_columns].mean(axis=1) * 0.4
        parallel_avg = merged_df[parallel_columns].mean(axis=1) * 0.3
        multi_avg = merged_df[multi_columns].mean(axis=1) * 0.3

        merged_df['weighted_average_score'] = origin_avg + parallel_avg + multi_avg
    else:
        # 如果没有任何得分列nn
        merged_df['weighted_average_score'] = pd.Series([0] * len(merged_df))

    return merged_df.iloc[:, [0, -1]]


class NeedleBenchSummarizer:
    """NeedleBench summarizer in OpenCompass.

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
                                tmp_scores = {metric: {k.split('@')[0]: v for k, v in scores[metric].items()} for metric in scores}
                                numerator = sum(tmp_scores[metric][k] * sg['weights'][k] for k in sg['weights'] if sg['weights'][k] != 0)
                            denominator = sum(sg['weights'].values())
                        else:
                            numerator = sum(scores[metric].values())
                            denominator = len(scores[metric])
                        scores[metric] = result[metric] = numerator / denominator
                    eval_modes = list(set(eval_modes))
                    eval_mode = eval_modes[0] if len(eval_modes) == 1 else 'mixed'

                # add to global results
                raw_results[model_abbr].setdefault(sg['name'], {}).update(scores)
                parsed_results[model_abbr].setdefault(sg['name'], {}).update(result)
                dataset_metrics.setdefault(sg['name'], []).extend(group_metrics)
                dataset_eval_mode[sg['name']] = eval_mode

        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode

    def _format_table(self, parsed_results, dataset_metrics, dataset_eval_mode):
        dataset_abbrs = [dataset_abbr_from_cfg(dataset) for dataset in self.dataset_cfgs]
        prompt_version = {dataset_abbr_from_cfg(d): get_prompt_hash(d)[:6] for d in self.dataset_cfgs}

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

        for key in dataset_metrics:
            dataset_metrics[key] = list(set(dataset_metrics[key]))

        for dataset_abbr, metric in summarizer_dataset_abbrs:
            if dataset_abbr not in dataset_metrics:
                table.append([dataset_abbr, '-', '-', '-'] + ['-'] * len(self.model_abbrs))
                table.append(header)
                continue
            if len(dataset_metrics[dataset_abbr]) >= 10:
                metric = 'average_score'

            if metric is None:
                metric = dataset_metrics[dataset_abbr][0]
            elif metric in dataset_metrics[dataset_abbr]:
                pass
            else:
                table.append([dataset_abbr, '-', '-', '-'] + ['-'] * len(self.model_abbrs))
                continue

            row = [dataset_abbr, prompt_version.get(dataset_abbr, '-'), metric, dataset_eval_mode.get(dataset_abbr, '-')]
            for model_abbr in self.model_abbrs:
                if dataset_abbr in parsed_results[model_abbr]:
                    row.append('{:.02f}'.format(parsed_results[model_abbr][dataset_abbr][metric]))
                else:
                    row.append('-')

            table.append(row)
        for i in range(len(table)):
            if i == 0 or table[i][0].startswith('---------'):
                table[i] = [table[i][0]] + table[i][4:]
            else:
                table[i] = [table[i][0]] + table[i][4:]

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

    def _read_and_sort_dataframe(self, file_path):
        # Read the file without treating the first row as a header
        df = pd.read_csv(file_path, header=None)

        # Function to sort columns based on the value of a specific row, excluding the first column
        def sort_columns_based_on_row_corrected(df, base_row_idx, start_row_idx, end_row_idx):
            # Extract the rows for sorting
            sort_values_row = df.iloc[base_row_idx, 1:].replace('-', np.nan).apply(pd.to_numeric, errors='coerce')
            # Handle NaNs by setting them to a value less than the minimum or using a method to keep them at the end
            min_possible_value = sort_values_row.min(skipna=True) - 1  # Use min value in the row minus 1 or another method
            sort_values_row_filled = sort_values_row.fillna(min_possible_value)
            # Get the sorted order of indices, excluding the first column
            sorted_col_indices = sort_values_row_filled.sort_values(ascending=False).index
            # Apply the sorted column indices to the whole DataFrame, adjusting for Python's 0-based index
            df.iloc[start_row_idx:end_row_idx+1] = df.iloc[start_row_idx:end_row_idx+1, [0] + sorted_col_indices.tolist()]

        # Apply the corrected sorting function based on the description
        sort_columns_based_on_row_corrected(df, 1, 0, 2)  # For rows 1-2 based on row 2's values
        sort_columns_based_on_row_corrected(df, 4, 3, 7)  # For rows 4-7 based on row 5's values
        sort_columns_based_on_row_corrected(df, 9, 8, 12)  # For rows 9-12 based on row 10's values
        sort_columns_based_on_row_corrected(df, 14, 13, 25)  # For rows 14-25 based on row 15's values

        # Return the sorted DataFrame
        return df

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
                    tabulate.tabulate(table, headers='firstrow') + '\n' + \
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

        # 读取、排序并获取DataFrame
        df_sorted = self._read_and_sort_dataframe(output_csv_path)

        # 导出排序后的DataFrame为CSV文件
        sorted_file_path = osp.abspath(output_csv_path).split('.')[0] + '_sorted.csv'  # 指定输出文件的路径
        df_sorted.to_csv(sorted_file_path, index=False, header=False)

        self.logger.info(f'write sorted csv to {sorted_file_path}')


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
        table = self._format_table(parsed_results, dataset_metrics, dataset_eval_mode)

        # format raw txt
        raw_txts = self._format_raw_txt(raw_results)

        # output to screen
        print(tabulate.tabulate(table, headers='firstrow'))

        # output to .text / .csv files
        self._output_to_file(output_path, time_str, table, raw_txts)

        if self.lark_reporter:
            content = f'{getpass.getuser()} 的'
            content += f'详细评测汇总已输出至 {osp.abspath(output_path)}'
            self.lark_reporter.post(content)

        if output_path is None:
            output_path = osp.join(self.work_dir, 'summary', f'summary_{time_str}.txt')
        # plot to show visualize results
        save_results_to_plots(output_path)


class NeedleBenchATCSummarizer:
    """NeedleBench summarizer in OpenCompass.

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
                                tmp_scores = {metric: {k.split('@')[0]: v for k, v in scores[metric].items()} for metric in scores}
                                numerator = sum(tmp_scores[metric][k] * sg['weights'][k] for k in sg['weights'] if sg['weights'][k] != 0)
                            denominator = sum(sg['weights'].values())
                        else:
                            numerator = sum(scores[metric].values())
                            denominator = len(scores[metric])
                        scores[metric] = result[metric] = numerator / denominator
                    eval_modes = list(set(eval_modes))
                    eval_mode = eval_modes[0] if len(eval_modes) == 1 else 'mixed'

                # add to global results
                raw_results[model_abbr].setdefault(sg['name'], {}).update(scores)
                parsed_results[model_abbr].setdefault(sg['name'], {}).update(result)
                dataset_metrics.setdefault(sg['name'], []).extend(group_metrics)
                dataset_eval_mode[sg['name']] = eval_mode

        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode

    def _format_table(self, parsed_results, dataset_metrics, dataset_eval_mode):
        dataset_abbrs = [dataset_abbr_from_cfg(dataset) for dataset in self.dataset_cfgs]
        prompt_version = {dataset_abbr_from_cfg(d): get_prompt_hash(d)[:6] for d in self.dataset_cfgs}

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

        for key in dataset_metrics:
            dataset_metrics[key] = list(set(dataset_metrics[key]))

        for dataset_abbr, metric in summarizer_dataset_abbrs:
            if dataset_abbr not in dataset_metrics:
                table.append([dataset_abbr, '-', '-', '-'] + ['-'] * len(self.model_abbrs))
                table.append(header)
                continue
            if len(dataset_metrics[dataset_abbr]) >= 10:
                metric = 'average_score'

            if metric is None:
                metric = dataset_metrics[dataset_abbr][0]
            elif metric in dataset_metrics[dataset_abbr]:
                pass
            else:
                table.append([dataset_abbr, '-', '-', '-'] + ['-'] * len(self.model_abbrs))
                continue

            row = [dataset_abbr, prompt_version.get(dataset_abbr, '-'), metric, dataset_eval_mode.get(dataset_abbr, '-')]
            for model_abbr in self.model_abbrs:
                if dataset_abbr in parsed_results[model_abbr]:
                    row.append('{:.02f}'.format(parsed_results[model_abbr][dataset_abbr][metric]))
                else:
                    row.append('-')

            table.append(row)
        for i in range(len(table)):
            if i == 0 or table[i][0].startswith('---------'):
                table[i] = [table[i][0]] + table[i][4:]
            else:
                table[i] = [table[i][0]] + table[i][4:]

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

    def _read_and_sort_dataframe(self, file_path):
        # Read the file without treating the first row as a header
        data = pd.read_csv(file_path)
        # print(data)
        # Correct the extraction of needle counts for all settings
        data['needle_count'] = data['dataset'].str.extract(r'needle_(\d+)_').astype(float)
        data['needle_count'] = data['needle_count'].astype(int)

        # Define experimental settings groups
        experimental_settings = {
            'en': '_en$',
            'zh': '_zh$',
            'en_ordered': '_en_ordered',
            'zh_ordered': '_zh_ordered',
        }

        # Function to calculate maximum needles
        def calculate_max_needles(dataset):
            max_needles = {model: None for model in dataset.columns if 'b' in model}
            for model in max_needles.keys():
                consecutive_low_scores = 0
                previous_needle_count = 0
                for index, row in dataset.sort_values(by='needle_count').iterrows():
                    try:
                        score = float(row[model])
                    except ValueError as e:
                        score = -1
                    if score < 60:
                        consecutive_low_scores += 1
                        if consecutive_low_scores == 1:
                            max_needles[model] = previous_needle_count
                    else:
                        consecutive_low_scores = 0
                    previous_needle_count = row['needle_count']
                max_needle_count_seen = dataset['needle_count'].max()
                max_needles[model] = max_needle_count_seen if max_needles[model] is None else max_needles[model]
            return max_needles

        # Calculate max needles for each group and organize results in a DataFrame
        results = {}
        for setting, regex in experimental_settings.items():
            filtered_data = data[data['dataset'].str.contains(regex)]
            results[setting] = calculate_max_needles(filtered_data)

        # Convert results to DataFrame and transpose it
        results_df = pd.DataFrame(results).transpose()

        # Return the sorted DataFrame
        results_df.index.name = 'ATC Experiment Type'
        return results_df

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
                    tabulate.tabulate(table, headers='firstrow') + '\n' + \
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
        # self.logger.info(f'write csv to {osp.abspath(output_csv_path)}')

        # 读取、排序并获取DataFrame
        df_sorted = self._read_and_sort_dataframe(output_csv_path)

        # 导出排序后的DataFrame为CSV文件
        df_sorted.to_csv(output_csv_path)

        self.logger.info(f'write sorted csv to {output_csv_path}')


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
        table = self._format_table(parsed_results, dataset_metrics, dataset_eval_mode)

        # format raw txt
        raw_txts = self._format_raw_txt(raw_results)

        # output to .text / .csv files
        self._output_to_file(output_path, time_str, table, raw_txts)

        if self.lark_reporter:
            content = f'{getpass.getuser()} 的'
            content += f'详细评测汇总已输出至 {osp.abspath(output_path)}'
            self.lark_reporter.post(content)

        if output_path is None:
            output_path = osp.join(self.work_dir, 'summary', f'summary_{time_str}.txt')
