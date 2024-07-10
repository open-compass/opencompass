# flake8: noqa
import json
import os.path as osp

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class FofoDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        filename = osp.join(path, f'{name}.json')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                question = problem['instruction']
                lan = 'cn' if 'cn' in name else 'en'
                raw_data.append({
                    'question': question,
                    'judge': {
                        'lan': lan,
                        'id': problem['id'],
                        'domain': problem['domain'],
                        'sub_domain': problem['sub_domain'],
                        'format': problem['format'],
                        'format_type': problem['format_type'],
                        'question': question
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
