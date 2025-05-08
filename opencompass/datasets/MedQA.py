from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MedQADataset(BaseDataset):

    @staticmethod
    def load_single(path):
        dataset = []
        ds = load_dataset(path)
        for data in ds['train']:
            data['label'] = data['answer_idx']
            choices = ''
            for option in data['options']:
                choices += option + '. ' + data['options'][option] + '\n'
            data['choices'] = choices

            dataset.append(data)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        dataset = MedQADataset.load_single(path)
        return dataset
