# flake8: noqa
import json
import os.path as osp

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class CompassBenchCheklistDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        filename = osp.join(path, f'{name}.json')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                question = problem['instruction']
                checklist_mardkdown = ''
                if problem.get('checklist', None):
                    for checklist_item in problem['checklist']:
                        checklist_mardkdown += f'- {checklist_item}\n'
                raw_data.append({
                    'question': question,
                    'checklist': checklist_mardkdown,
                    'judge': {
                        'category': problem.get('category', None),
                        'lan': problem.get('lan', None),
                        'id': problem.get('id', None),
                        'question': question
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
