from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset
from .utils import iter_jsonl


@LOAD_DATASET.register_module()
class InfiniteBenchretrievepasskeyDataset(BaseDataset):

    @staticmethod
    def load(path: str):

        dataset = list(iter_jsonl(path))

        raw_data = []
        for item in dataset:
            context = item['context']
            input = item['input']
            answer = item['answer']
            raw_data.append({
                'context': context,
                'input': input,
                'answer': answer
            })
        dataset = Dataset.from_list(raw_data)
        return dataset
