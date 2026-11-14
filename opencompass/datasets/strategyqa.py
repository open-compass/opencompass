import json
import re
from os import environ

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset


@TEXT_POSTPROCESSORS.register_module('strategyqa')
def strategyqa_pred_postprocess(text: str) -> str:
    # An explicit "answer is yes/no" is the verdict; without one, the first
    # yes or no in the output is, since later ones sit in the reasoning.
    markers = list(
        re.finditer(r'answer\s+is\s*[:,]?\s*(yes|no)(?![a-z])', text,
                    re.IGNORECASE))
    if markers:
        return markers[-1].group(1).lower()
    match = re.search(r'(?<![a-z])(yes|no)(?![a-z])', text.lower())
    if match:
        return match.group(1)
    return ''


@TEXT_POSTPROCESSORS.register_module('strategyqa_dataset')
def strategyqa_dataset_postprocess(text: str) -> str:
    return 'yes' if str(text) == 'True' else 'no'


@LOAD_DATASET.register_module()
class StrategyQADataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)

        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(path, split='train')
        else:
            with open(path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            dataset = Dataset.from_list(dataset)
        return dataset
