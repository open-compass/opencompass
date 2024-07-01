# flake8: noqa
# yapf: disable

from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


def _parse(item):
    choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    s = ''
    for i, opt in enumerate(item['options']):
        if opt == 'N/A':
            continue
        s += '{}. {}\n'.format(choices[i], opt)
    item['options_str'] = s.strip()
    item['cot_content'] = item['cot_content'].removeprefix("A: Let's think step by step.").strip()
    return item


@LOAD_DATASET.register_module()
class MMLUProDataset(BaseDataset):

    @staticmethod
    def load(category: str):
        mmlu_pro = load_dataset('TIGER-Lab/MMLU-Pro')
        mmlu_pro = mmlu_pro.filter(lambda x: x['category'] == category)
        mmlu_pro = mmlu_pro.map(_parse)
        return mmlu_pro
