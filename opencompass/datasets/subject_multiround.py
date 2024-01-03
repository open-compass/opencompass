# flake8: noqa: E501
import json
import os.path as osp
import re
from typing import Optional

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

eng_base_prefix = """
"""
chn_base_prefix = """
"""


def prompt_construct(sample):
    prompt, suffix = '', ''
    return prompt, suffix


@LOAD_DATASET.register_module()
class MultiroundDataset(BaseDataset):

    def load(
        self,
        path: str,
        name: str,
    ):
        filename = osp.join(path, f'{name}.json')
        dataset = DatasetDict()
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                dialogue = problem['dialogue']
                capability = str(problem['capability'])
                others = problem['others']
                raw_data.append({
                    'dialogue': dialogue,
                    'capability': capability,
                    'gpt4_prefix': '',
                    'gpt4_suffix': '',
                    'others': others,
                    'judge': {
                        'capability': capability
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
