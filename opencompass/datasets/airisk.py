from datasets import load_dataset

from .base import BaseDataset


class AiRiskDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        """Load dataset and aug with original dataset."""

        dataset = load_dataset('json', data_files=path)

        def choices_process(example):
            for i in 'AB':
                if i in example['answer_matching_behavior']:
                    example['answer_matching_behavior'] = i
                    break
            return example

        dataset = dataset.map(choices_process)
        return dataset
