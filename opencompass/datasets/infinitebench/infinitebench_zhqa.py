from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import iter_jsonl


@LOAD_DATASET.register_module()
class InfiniteBenchzhqaDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)

        dataset = list(iter_jsonl(path))

        raw_data = []
        for item in dataset:
            context = item['context']
            question = item['input']
            answer = item['answer']
            raw_data.append({
                'context': context,
                'question': question,
                'answer': answer
            })
        dataset = Dataset.from_list(raw_data)
        return dataset
