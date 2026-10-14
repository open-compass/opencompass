#! /usr/bin/env python

from pathlib import Path

import yaml
from tabulate import tabulate

OC_ROOT = Path(__file__).absolute().parents[2]
DOC_ROOT = Path(__file__).absolute().parent
GITHUB_PREFIX = 'https://github.com/open-compass/opencompass/tree/main/'
statistics_fragment = DOC_ROOT / 'dataset_statistics.inc'

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
                    if i[j][index]:
                        table_format_list_sub.append('[link](' + i[j][index] +
                                                     ')')
                    else:
                        table_format_list_sub.append('')
                elif index == 'configpath_llmjudge':
                    if i[j][index] == '':
                        table_format_list_sub.append(i[j][index])
                    elif isinstance(i[j][index], list):
                        sub_list_text = ''
                        for k in i[j][index]:
                            sub_list_text += (link_token + GITHUB_PREFIX + k +
                                              ') / ')
                        table_format_list_sub.append(sub_list_text[:-2])
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
    table_cfg = dict(tablefmt='pipe',
                     floatfmt='.2f',
                     numalign='right',
                     stralign='center')
    header = [
        'Name', 'Category', 'Paper or Repository', 'Recommended Config',
        'Recommended Config (LLM Judge)'
    ]
    table = tabulate(data_list, header, **table_cfg)

    # The fragment is embedded in user_guides/datasets.md. The .inc suffix
    # prevents Sphinx from treating it as a standalone document.
    with open(statistics_fragment, 'w') as f:
        f.write("""```{table}\n:class: dataset\n""")
        f.write(table)
        f.write('\n```\n')


generate_table(data_list=data_format_list)
