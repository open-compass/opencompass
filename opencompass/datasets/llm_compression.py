import os.path as osp
from typing import List

from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class LLMCompressionDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: List[str] = None, samples: int = None):

        # Check if file exists in the given path
        supported_extensions = ['json', 'jsonl']
        for ext in supported_extensions:
            filename = osp.join(
                path, f'{name}.{ext}')  # name refers to data subset name

            if osp.exists(filename):
                break
        else:
            raise FileNotFoundError(f'{filename} not found.')

        samples = 'test' if samples is None else f'test[:{samples}]'

        data_files = {'test': filename}
        dataset = load_dataset('json', data_files=data_files, split=samples)

        # Filter out empty samples
        dataset = dataset.filter(lambda example: len(example['content']) > 0)

        return dataset
