import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CBDatasetV2(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['label'] = {
                    'contradiction': 'A',
                    'entailment': 'B',
                    'neutral': 'C'
                }[line['label']]
                dataset.append(line)
        return Dataset.from_list(dataset)
