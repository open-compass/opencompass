from datasets import DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MMLUCFDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        """Loading HuggingFace datasets."""
        # Use HuggingFace's load_dataset method to load the dataset
        hf_dataset = load_dataset(path)
        columns_to_keep = ['Question', 'A', 'B', 'C', 'D', 'Answer']
        hf_dataset = hf_dataset.map(
            lambda x: {key: x[key]
                       for key in columns_to_keep})
        splits = ['dev', 'val']

        for split in splits:
            sub_set = f'{name}_{split}'

            # Rename fields here if they don't match the expected names
            hf_dataset[sub_set] = hf_dataset[sub_set].map(
                lambda example: {
                    'input': example['Question'],
                    'A': example['A'],
                    'B': example['B'],
                    'C': example['C'],
                    'D': example['D'],
                    'target': example['Answer']
                })

        # Create a DatasetDict and return it
        dataset = DatasetDict({
            'dev': hf_dataset[f'{name}_{splits[0]}'],
            'test': hf_dataset[f'{name}_{splits[1]}']  # Use 'val' as 'test'
        })
        return dataset
