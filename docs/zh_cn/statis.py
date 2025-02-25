#! /usr/bin/env python

from pathlib import Path

import yaml
from tabulate import tabulate

OC_ROOT = Path(__file__).absolute().parents[2]
GITHUB_PREFIX = 'https://github.com/open-compass/opencompass/tree/main/'
DATASETZOO_TEMPLATE = """\
# 数据集统计

在本页面中，我们列举了OpenCompass所支持的所有数据集。

你可以使用排序和搜索功能找到需要的数据集。

"""

with open('dataset_statistics.md', 'w') as f:
    f.write(DATASETZOO_TEMPLATE)

load_path = str(OC_ROOT / 'dataset-index.yml')

with open(load_path, 'r') as f2:
    data_list = yaml.load(f2, Loader=yaml.FullLoader)

HEADER = ['name', 'category', 'paper', 'configpath']


def table_format(data_list):
    table_format_list = []
    for i in data_list:
        table_format_list_sub = []
        for j in i:
            for index in HEADER:
                if index == 'paper':
                    table_format_list_sub.append('[链接](' + i[j][index] + ')')
                elif index == 'configpath':
                    if isinstance(i[j][index], list):
                        sub_list_text = ''
                        for k in i[j][index]:
                            sub_list_text += ('[链接](' + GITHUB_PREFIX + k +
                                              ') / ')
                        table_format_list_sub.append(sub_list_text[:-2])
                    else:
                        table_format_list_sub.append('[链接](' + GITHUB_PREFIX +
                                                     i[j][index] + ')')
                else:
                    table_format_list_sub.append(i[j][index])
        table_format_list.append(table_format_list_sub)
    return table_format_list


data_format_list = table_format(data_list)


def generate_table(data_list, title=None):

    with open('dataset_statistics.md', 'a') as f:
        if title is not None:
            f.write(f'\n{title}')
        f.write("""\n```{table}\n:class: dataset\n""")
        header = ['数据集名称', '数据集类型', '原文或资源地址', '配置文件链接']
        table_cfg = dict(tablefmt='pipe',
                         floatfmt='.2f',
                         numalign='right',
                         stralign='center')
        f.write(tabulate(data_list, header, **table_cfg))
        f.write('\n```\n')


generate_table(
    data_list=data_format_list,
    title='## 支持数据集列表',
)
