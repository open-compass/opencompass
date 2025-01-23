import re
import warnings
from collections import defaultdict
from pathlib import Path

from modelindex.load_model_index import load
from modelindex.models.Result import Result
from tabulate import tabulate
import yaml

OC_ROOT = Path(__file__).absolute().parents[2]
PAPERS_ROOT = Path('papers')  # Path to save generated paper pages.
GITHUB_PREFIX = 'https://github.com/open-compass/opencompass/blob/main'
DATASETZOO_TEMPLATE = """\
# 数据集统计

在本页面中，我们列举了OpenCompass所支持的所有数据集。

你可以使用排序和搜索功能找到需要的数据集。

"""

with open('dataset_statistics.md', 'w') as f:
    f.write(DATASETZOO_TEMPLATE)
    f.close()

load_path = str(OC_ROOT / 'dataset-index.yml')

with open(load_path, 'r') as f2:
    data_list = yaml.load(f2, Loader=yaml.FullLoader)

HEADER = ['name', 'category']

def table_format(data_list):
    table_format_list = []
    for i in data_list:
        print(i)
        table_format_list_sub = []
        for j in i:
            print(i[j])
            for index in HEADER:
                table_format_list_sub.append(i[j][index])
        table_format_list.append(table_format_list_sub)
    return table_format_list

data_format_list = table_format(data_list)


def generate_table(data_list, title=None):

    with open('dataset_statistics.md', 'a') as f:
        if title is not None:
            f.write(f'\n{title}')
        f.write("""\n```{table}\n:class: dataset\n""")
        header = [
            '数据集名称',
            '数据集类型',
        ]
        table_cfg = dict(
            tablefmt='pipe',
            floatfmt='.2f',
            numalign='right',
            stralign='center')
        f.write(tabulate(data_list, header, **table_cfg))
        f.write('\n```\n')


generate_table(
    data_list=data_format_list,
    title='## 支持数据集列表',
)

# breakpoint()
#
#
#
#
# model_index = load(str(MMPT_ROOT / 'dataset-index.yml'))
#
#
#
# def scatter_results(models):
#     model_result_pairs = []
#     for model in models:
#         if model.results is None:
#             result = Result(task=None, dataset=None, metrics={})
#             model_result_pairs.append((model, result))
#         else:
#             for result in model.results:
#                 model_result_pairs.append((model, result))
#     return model_result_pairs
#
#
# def generate_summary_table(task, model_result_pairs, title=None):
#     metrics = set()
#     for model, result in model_result_pairs:
#         if result.task == task:
#             metrics = metrics.union(result.metrics.keys())
#     metrics = sorted(list(metrics))
#
#     rows = []
#     for model, result in model_result_pairs:
#         if result.task != task:
#             continue
#         name = model.name
#         params = f'{model.metadata.parameters / 1e6:.2f}'  # Params
#         if model.metadata.flops is not None:
#             flops = f'{model.metadata.flops / 1e9:.2f}'  # Flops
#         else:
#             flops = None
#         readme = Path(model.collection.filepath).parent.with_suffix('.md').name
#         page = f'[链接]({PAPERS_ROOT / readme})'
#         model_metrics = []
#         for metric in metrics:
#             model_metrics.append(str(result.metrics.get(metric, '')))
#
#         rows.append([name, params, flops, *model_metrics, page])
#
#     with open('modelzoo_statistics.md', 'a') as f:
#         if title is not None:
#             f.write(f'\n{title}')
#         f.write("""\n```{table}\n:class: model-summary\n""")
#         header = [
#             '模型',
#             '参数量 (M)',
#             'Flops (G)',
#             *[METRIC_ALIAS.get(metric, metric) for metric in metrics],
#             'Readme',
#         ]
#         table_cfg = dict(
#             tablefmt='pipe',
#             floatfmt='.2f',
#             numalign='right',
#             stralign='center')
#         f.write(tabulate(rows, header, **table_cfg))
#         f.write('\n```\n')
#
#
# def generate_dataset_wise_table(task, model_result_pairs, title=None):
#     dataset_rows = defaultdict(list)
#     for model, result in model_result_pairs:
#         if result.task == task:
#             dataset_rows[result.dataset].append((model, result))
#
#     if title is not None:
#         with open('modelzoo_statistics.md', 'a') as f:
#             f.write(f'\n{title}')
#     for dataset, pairs in dataset_rows.items():
#         generate_summary_table(task, pairs, title=f'### {dataset}')
#
#
# model_result_pairs = scatter_results(model_index.models)
#
# # Generate Pretrain Summary
# generate_summary_table(
#     task=None,
#     model_result_pairs=model_result_pairs,
#     title='## 预训练模型',
# )
#
# # Generate Image Classification Summary
# generate_dataset_wise_table(
#     task='Image Classification',
#     model_result_pairs=model_result_pairs,
#     title='## 图像分类',
# )
#
# # Generate Multi-Label Classification Summary
# generate_dataset_wise_table(
#     task='Multi-Label Classification',
#     model_result_pairs=model_result_pairs,
#     title='## 图像多标签分类',
# )
#
# # Generate Image Retrieval Summary
# generate_dataset_wise_table(
#     task='Image Retrieval',
#     model_result_pairs=model_result_pairs,
#     title='## 图像检索',
# )