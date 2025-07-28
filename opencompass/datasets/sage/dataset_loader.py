from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class SAGEDataset(BaseDataset):

    @staticmethod
    def load(split: str = 'val'):
        hf_dataset = load_dataset('opencompass/SAGE', split=split)
        dataset = []
        for example in hf_dataset:
            dataset.append({
                'problem': example['question'],
                'answer': example.get('standard_answer', '')
            })
        return Dataset.from_list(dataset)
