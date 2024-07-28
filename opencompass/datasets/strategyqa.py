import json
import re
from os import environ

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset


@TEXT_POSTPROCESSORS.register_module('strategyqa')
def strategyqa_pred_postprocess(text: str) -> str:
    text = text.split('\n\n')[0]
    text = text.split('answer is ')[-1]
    match = re.search(r'(yes|no)', text.lower())
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
            dataset = MsDataset.load('opencompass/strategy_qa',
                                     split='train',
                                     trust_remote_code=True)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            dataset = Dataset.from_list(dataset)
        return dataset
