# flake8: noqa: E501
# yapf: disable
import copy
import json
import os.path as osp
import re
import tempfile
from os import environ
from typing import List
import os
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

HUMANEVAL_IMPORT_ERROR = '''\
Please install human_eval use following steps:
git clone git@github.com:open-compass/human-eval.git
cd human-eval && pip install -e .'''

HUMANEVAL_PLUS_IMPORT_ERROR = '''\
Please install evalplus use following steps:
git clone --recurse-submodules git@github.com:open-compass/human-eval.git
cd human-eval
pip install -e .
pip install -e evalplus'''


@LOAD_DATASET.register_module()
class xHumanevalDataset(BaseDataset):

    @staticmethod
    def load(path: str, num_repeats: int = 1, local_mode: bool = False):
        """Load humaneval dataset for pass k mode.

        Note that you can use num_repeats > 1 when your model does not support
        `num_return_sequence` in generation, otherwise use the raw
        humaneval dataset and set `num_return_sequence` in model config to
        generate multiple responses for testing pass@k>1.

        It better to change your dataset abbr correspondingly if you want to
        change num_repeats>1, otherwise the number in
        `.cache/dataset_size.json` might be inconsistent.

        Args:
            num_repeats(int): Number of repetition for this dataset to get
        multiple responses in special cases.
        """
        path = get_data_path(path, local_mode=True)
        # path = os.path.join(path, name)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(path, subset_name='openai_humaneval', split='test')
            dataset_list = []
            for example in dataset:
                dataset_list.extend([example] * num_repeats)
            dataset = Dataset.from_list(dataset_list)
        else:
            dataset = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    item = {'prompt': item['prompt'], 'task_id': item['task_id'], 'test': item['test'], 'entry_point': item['entry_point'], 'canonical_solution': item['canonical_solution']}
                    dataset.extend(
                        [item for _ in range(num_repeats)])
            print(dataset[:10])
            dataset = Dataset.from_list(dataset)
        return dataset


