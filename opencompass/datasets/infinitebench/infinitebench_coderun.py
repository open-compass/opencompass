import re

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import iter_jsonl


@LOAD_DATASET.register_module()
class InfiniteBenchcoderunDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)

        dataset = list(iter_jsonl(path))

        raw_data = []
        for item in dataset:
            context = item['context']
            find_result = re.findall(r'func_[0-9]+\(\-?[0-9]+\)',
                                     item['input'])
            func_call = find_result[0]
            func = func_call.split('(')[0]
            answer = item['answer']
            raw_data.append({
                'context': context,
                'func': func,
                'func_call': func_call,
                'answer': answer
            })
        dataset = Dataset.from_list(raw_data)
        return dataset
