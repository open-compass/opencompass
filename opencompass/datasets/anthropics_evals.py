from datasets import load_dataset

from .base import BaseDataset


class AiRiskDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        """Load dataset."""

        dataset = load_dataset('json', data_files=path)

        def choices_process(example):
            # the original answer format is ` (A)`, etc.
            for i in 'ABCDEFGH':
                if i in example['answer_matching_behavior']:
                    example['answer_matching_behavior'] = i
                    break
            return example

        dataset = dataset.map(choices_process)
        return dataset


class PersonaDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        """Load dataset."""

        dataset = load_dataset('json', data_files=path)

        def choices_process(example):
            # the original answer format is ` No` or ` Yes`.
            if example['answer_matching_behavior'] == ' Yes':
                example['answer_matching_behavior'] = 'A'
            else:
                example['answer_matching_behavior'] = 'B'
            return example

        dataset = dataset.map(choices_process)
        return dataset


class SycophancyDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        """Load dataset."""

        dataset = load_dataset('json', data_files=path)

        def choices_process(example):
            # the original answer format is ` (A)`, etc.
            for i in 'ABCDEFG':
                if i in example['answer_matching_behavior']:
                    example['answer_matching_behavior'] = i
                    break
            return example

        dataset = dataset.map(choices_process)
        return dataset
