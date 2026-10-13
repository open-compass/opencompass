# flake8: noqa
import json
import os.path as osp
import re

import numpy as np
import pandas as pd
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (DICT_POSTPROCESSORS, ICL_EVALUATORS,
                                  LOAD_DATASET)
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class JudgeBenchDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):

        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                conversation_a = item['chosen']
                conversation_b = item['rejected']
                model_a = item['chosen_model']
                model_b = item['rejected_model']
                question = item['prompt']
                winner = item['winner']
                if winner == 'B':
                    conversation_a, conversation_b = conversation_b, conversation_a
                    model_a, model_b = model_b, model_a
                subset = item['subset']
                lan = 'en'
                raw_data.append({
                    'question': question,
                    'answerA': conversation_a,
                    'answerB': conversation_b,
                    'judge': {
                        'prompt': item['prompt'],
                        'Answer_A': conversation_a,
                        'Answer_B': conversation_b,
                        'subset': subset,
                        'winner': winner,
                        'model_a': model_a,
                        'model_b': model_b,
                        'dataset_name': 'rewardbench',
                        'lan': lan
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
