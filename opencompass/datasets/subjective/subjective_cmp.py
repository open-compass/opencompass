import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class SubjectiveCmpDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.json')
        dataset = DatasetDict()
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                question = problem['question']
                capability = problem['capability']
                others = problem['others']
                raw_data.append({
                    'question': question,
                    'capability': capability,
                    'others': others,
                    'judge': {
                        'capability': capability,
                        'question': question
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
