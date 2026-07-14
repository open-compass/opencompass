import json

from datasets import Dataset

from opencompass.utils import get_data_path  # noqa: F401, F403

from .base import BaseDataset


class OJBenchDataset(BaseDataset):

    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset_list = []
        li = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                data = json.loads(line)
                dataset_list.append({
                    'id': str(data['id']),
                    'prompt': data['prompt']
                })
                li += 1
        return Dataset.from_list(dataset_list)
