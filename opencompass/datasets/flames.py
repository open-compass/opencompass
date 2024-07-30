# flake8: noqa: E501
import json
import os.path as osp
import re
from typing import Optional

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .subjective.subjective_cmp import SubjectiveCmpDataset


class Config:

    def __init__(self, flames_config_path, flames_bench_config_name) -> None:
        config_file_path = osp.join(flames_config_path,
                                    flames_bench_config_name)
        with open(config_file_path, 'r') as config_file:
            self.config = ''.join(config_file.readlines())
            config_file.close()


def prompt_construct(sample, config: Config):
    dimensions = config.config
    base_prompt = '{dimensions}'\
        '{question}\n' \
        '回答: '
    prompt = base_prompt.format(dimensions=dimensions,
                                question=sample['prompt'])

    return prompt


@LOAD_DATASET.register_module()
class FlamesDataset(SubjectiveCmpDataset):

    def load(
        self,
        path: str,
        name: str,
    ):
        path = get_data_path(path, local_mode=True)
        config = Config(path, f'{name}_config.txt')

        dataset = []
        with open(osp.join(path, f'{name}.json')) as f:
            dataset = json.load(f)
        flames_dataset = []
        for ins in dataset:
            ins['instruction'] = prompt_construct(ins, config)
            ins['judge'] = {
                'dimension': ins['dimension'],
                'subcomponent': ins['subcomponent']
            }
            flames_dataset.append(ins)
        flames_dataset = Dataset.from_list(flames_dataset)
        return flames_dataset
