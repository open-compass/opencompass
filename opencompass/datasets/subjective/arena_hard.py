import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class ArenaHardDataset(BaseDataset):

    def load(self, path: str, name: str, *args, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.jsonl')
        dataset = DatasetDict()
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                problem = json.loads(line)
                question_id = problem['question_id']
                cluster = problem['cluster']
                question = problem['turns'][0][
                    'content']  # only one turn in arena_hard
                raw_data.append({
                    'question': question,
                    'capability': cluster,
                    'judge': {
                        'capability': cluster,
                        'question': question,
                        'question_id': question_id
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset
