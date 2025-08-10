from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import iter_jsonl


@LOAD_DATASET.register_module()
class InfiniteBenchcodedebugDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)

        dataset = list(iter_jsonl(path))

        raw_data = []
        for item in dataset:
            context = item['context']
            question = item['input']
            option_A = item['options'][0]
            option_B = item['options'][1]
            option_C = item['options'][2]
            option_D = item['options'][3]
            answer = chr(item['options'].index(item['answer'][0]) + ord('A'))
            raw_data.append({
                'context': context,
                'question': question,
                'option_A': option_A,
                'option_B': option_B,
                'option_C': option_C,
                'option_D': option_D,
                'answer': answer
            })
        dataset = Dataset.from_list(raw_data)
        return dataset
