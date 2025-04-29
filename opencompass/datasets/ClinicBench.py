from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ClinicBenchDataset(BaseDataset):

    @staticmethod
    def load_single():
        dataset = load_dataset('xuxuxuxuxu/Pharmacology-QA')['train']
        return dataset

    @staticmethod
    def load(path):
        train_dataset = Dataset.from_list([])
        val_dataset = ClinicBenchDataset.load_single()
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        return dataset
