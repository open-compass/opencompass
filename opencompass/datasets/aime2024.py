import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class Aime2024Dataset(BaseDataset):

    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                origin_prompt = line['origin_prompt']
                line['question'] = origin_prompt[:]
                line['answer'] = line['gold_answer']
                dataset.append(line)
        return Dataset.from_list(dataset)
