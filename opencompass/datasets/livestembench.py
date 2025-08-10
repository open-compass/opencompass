# Edited from the official SimpleQA config: https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py # noqa E501
import json
import os
import random

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class LiveStemBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str,
             num_examples: int | None = None,
             n_repeats: int = 1,
             version: str = 'livestembench-20241227',
             **kwargs):
        path = get_data_path(path)
        dataset = DatasetDict()
        path = os.path.join(path, f'{version}.json')
        with open(path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        pure_dataset = []
        for example in examples:
            if len(example['options']) > 0:
                example['question'] = example['question'] + '\n' + \
                    '\n'.join(
                    example['options'])
            pure_dataset.append({
                'question': example['question'],
                'answer': example['answer']
            })
        if num_examples:
            assert n_repeats == 1, \
                'n_repeats only supported when max_examples = None'
            rng = random.Random(0)
            pure_dataset = rng.sample(pure_dataset, num_examples)
        pure_dataset = pure_dataset * n_repeats
        dataset['train'] = Dataset.from_list(pure_dataset)
        dataset['test'] = Dataset.from_list(pure_dataset)
        return dataset
