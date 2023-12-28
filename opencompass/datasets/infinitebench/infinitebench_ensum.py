from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset
from .utils import iter_jsonl


@LOAD_DATASET.register_module()
class InfiniteBenchensumDataset(BaseDataset):

    @staticmethod
    def load(path: str):

        dataset = list(iter_jsonl(path))

        raw_data = []
        for item in dataset:
            context = item['context']
            answer = item['answer']
            raw_data.append({'context': context, 'answer': answer})
        dataset = Dataset.from_list(raw_data)
        return dataset
