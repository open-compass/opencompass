from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import get_single_answer, iter_jsonl


@LOAD_DATASET.register_module()
class InfiniteBenchretrievepasskeyDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)

        dataset = list(iter_jsonl(path))

        raw_data = []
        for item in dataset:
            context = item['context']
            input = item['input']
            answer = get_single_answer(item)
            raw_data.append({
                'context': context,
                'input': input,
                'answer': answer
            })
        dataset = Dataset.from_list(raw_data)
        return dataset
