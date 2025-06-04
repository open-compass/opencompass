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

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class RMBDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if item['subset'] == 'pair':
                    raw_data.extend(self.load_pair(item))
                elif item['subset'] == 'bon':
                    raw_data.extend(self.loadbon(item))
                else:
                    raise NotImplementedError
        dataset = Dataset.from_list(raw_data)
        return dataset

    def load_pair(self, item):
        raw_item_list = []
        conversation_a = item['chosen']['answer']
        conversation_b = item['reject']['answer']
        question = ''
        for line in item['conversation_input']:
            if line['role'] == 'user':
                question += '\n\n ### User:' + line['content']
            else:
                question += '\n\n ### Assistant:' + line['content']
        question += '\n\n ### Assistant:'
        winner = 'A'
        pair_uid = item['pair_uid']
        subset = item['subset']
        goal = item['goal']
        raw_item = {
            'question': question,
            'answerA': conversation_a,
            'answerB': conversation_b,
            'judge': {
                'question': question,
                'Answer_A': conversation_a,
                'Answer_B': conversation_b,
                'winner': winner,
                'pair_uid': pair_uid,
                'subset': subset,
                'goal': goal,
            }
        }
        raw_item_list.append(raw_item)
        return raw_item_list

    def loadbon(self, item):
        raw_item_list = []
        conversation_a = item['bon_best']['answer']
        question = ''
        for line in item['conversation_input']:
            if line['role'] == 'user':
                question += '\n\n ### User:' + line['content']
            else:
                question += '\n\n ### Assistant:' + line['content']
        question += '\n\n ### Assistant:'
        bon_uid = item['bon_uid']
        subset = item['subset']
        goal = item['goal']
        for loser in item['loser_list']:
            conversation_b = loser['answer']
            winner = 'A'
            raw_item = {
                'question': question,
                'answerA': conversation_a,
                'answerB': conversation_b,
                'judge': {
                    'question': question,
                    'Answer_A': conversation_a,
                    'Answer_B': conversation_b,
                    'winner': winner,
                    'bon_uid': bon_uid,
                    'subset': subset,
                    'goal': goal,
                }
            }
            raw_item_list.append(raw_item)
        return raw_item_list
