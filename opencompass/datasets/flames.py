# flake8: noqa: E501
import json
import os.path as osp
import re
from typing import Optional

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .subjective.subjective_cmp import SubjectiveCmpDataset


class Config:

    def __init__(self, alignment_bench_config_path,
                 alignment_bench_config_name) -> None:
        config_file_path = osp.join(alignment_bench_config_path,
                                    alignment_bench_config_name)
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

    def load(self,
             path: str,
             name: str,):
        config = Config(path, f'{name}_config.txt')


        dataset = []
        with open(f'{path}/{name}.json') as f:
            dataset = json.load(f)
        flames_dataset = []
        for ins in dataset:
            # prefix = prompt_construct(ins, config)
            # ins['prefix'] = prefix \
            # ins['reference'] = ins['answer']
            ins['instruction'] = prompt_construct(ins, config)
            ins['judge'] = {}
            flames_dataset.append(ins)
        flames_dataset = Dataset.from_list(flames_dataset)
        return flames_dataset


if __name__ == '__main__':
    data = {
        'question': '高音单簧管和高音萨克斯的调性相同吗？如果相同，请说出他们的调性，如果不同，请分别说出他们的调性',
        'capability': '专业能力',
        'others': {
            'subcategory': '音乐',
            'reference': '高音单簧管和高音萨克斯的调性不同。高音单簧管的调性通常为E♭，而高音萨克斯的调性则为B♭。\n',
            'question_id': 1
        }
    }
    prefix = prompt_construct(data, alignmentbench_config)
    print(prefix)
