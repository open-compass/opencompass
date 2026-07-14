from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class ATLASDataset(BaseDataset):

    @staticmethod
    def load(split: str = 'val'):
        hf_dataset = load_dataset('opencompass/ATLAS', split=split)
        dataset = []
        for example in hf_dataset:
            dataset.append({
                'problem':
                str(example['question']),
                'answer':
                str(example.get('refined_standard_answer', ''))
            })
        return Dataset.from_list(dataset)
