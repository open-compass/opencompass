import re

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import iter_jsonl


@LOAD_DATASET.register_module()
class InfiniteBenchmathfindDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)

        dataset = list(iter_jsonl(path))

        raw_data = []
        for item in dataset:
            context = item['context']
            answer = item['answer']
            find_result = re.findall(r'The .+ of', item['input'])
            target_number = find_result[0].lower()[:-3]
            prefix = f'What is {target_number} in the following list?'
            raw_data.append({
                'prefix': prefix,
                'context': context,
                'answer': answer
            })
        dataset = Dataset.from_list(raw_data)
        return dataset
