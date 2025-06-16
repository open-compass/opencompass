# flake8: noqa
import json
import os.path as osp
import re

import numpy as np
import pandas as pd
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class AiderDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for key, item in data.items():
                raw_data.append(self.process_item(key, item))
        dataset = Dataset.from_list(raw_data)
        return dataset

    def process_item(self, key, item):
        question = ''
        system_prompt = ''
        for line in item:
            if line['role'] == 'system':
                system_prompt = line['content']
            elif line['role'] == 'user':
                question += '\n\n ### User:' + line['content']
            else:
                question += '\n\n ### Assistant:' + line['content']
        question += '\n\n ### Assistant:'
        raw_item = {
            'system_prompt': system_prompt,
            'prompt': question,
            'judge': {
                'system_prompt': system_prompt,
                'prompt': question,
                'test_dir': key
            }
        }
        return raw_item
