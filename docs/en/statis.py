#! /usr/bin/env python

from pathlib import Path

import yaml
from tabulate import tabulate

OC_ROOT = Path(__file__).absolute().parents[2]
GITHUB_PREFIX = 'https://github.com/open-compass/opencompass/tree/main/'
DATASETZOO_TEMPLATE = """\
# Dataset Statistics

On this page, we have listed all the datasets supported by OpenCompass.

You can use sorting and search functions to find the dataset you need.

We provide recommended running configurations for each dataset,
and in some datasets also offer recommended configurations based on LLM Judge.

You can quickly start evaluation tasks based on the recommended configurations.
However, please note that these configurations may be updated over time.

"""

with open('dataset_statistics.md', 'w') as f:
    f.write(DATASETZOO_TEMPLATE)

load_path = str(OC_ROOT / 'dataset-index.yml')

with open(load_path, 'r') as f2:
    data_list = yaml.load(f2, Loader=yaml.FullLoader)

HEADER = ['name', 'category', 'paper', 'configpath', 'configpath_llmjudge']

recommanded_dataset_list = [
    'ifeval', 'aime2024', 'bbh', 'bigcodebench', 'cmmlu', 'drop', 'gpqa',
    'hellaswag', 'humaneval', 'korbench', 'livecodebench', 'math', 'mmlu',
    'mmlu_pro', 'musr', 'math500'
]


def table_format(data_list):
    table_format_list = []
    for i in data_list:
        table_format_list_sub = []
        for j in i:
            if j in recommanded_dataset_list:
                link_token = '[link]('
            else:
                link_token = '[link(TBD)]('

            for index in HEADER:
                if index == 'paper':
                    table_format_list_sub.append('[link](' + i[j][index] + ')')
                elif index == 'configpath_llmjudge':
                    if i[j][index] == '':
                        table_format_list_sub.append(i[j][index])
                    else:
                        table_format_list_sub.append(link_token +
                                                     GITHUB_PREFIX +
                                                     i[j][index] + ')')
                elif index == 'configpath':
                    if isinstance(i[j][index], list):
                        sub_list_text = ''
                        for k in i[j][index]:
                            sub_list_text += (link_token + GITHUB_PREFIX + k +
                                              ') / ')
                        table_format_list_sub.append(sub_list_text[:-2])
                    else:
                        table_format_list_sub.append(link_token +
                                                     GITHUB_PREFIX +
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
        header = [
            'Name', 'Category', 'Paper or Repository', 'Recommended Config',
            'Recommended Config (LLM Judge)'
        ]
        table_cfg = dict(tablefmt='pipe',
                         floatfmt='.2f',
                         numalign='right',
                         stralign='center')
        f.write(tabulate(data_list, header, **table_cfg))
        f.write('\n```\n')


generate_table(
    data_list=data_format_list,
    title='## Supported Dataset List',
)
