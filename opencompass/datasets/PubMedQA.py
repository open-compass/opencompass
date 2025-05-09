from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class PubMedQADataset(BaseDataset):

    @staticmethod
    def load_single(path):
        dataset = []
        ds = load_dataset(path, 'pqa_labeled')
        for data in ds['train']:
            data['question'] = (f"CONTEXTS: {data['context']}\n"
                                f"QUESTION: {data['question']}")
            choices = 'A. yes\nB. no\nC. maybe'
            data['choices'] = choices
            if data['final_decision'] == 'yes':
                data['label'] = 'A. yes'
            elif data['final_decision'] == 'no':
                data['label'] = 'B. no'
            else:
                data['label'] = 'C. maybe'

            dataset.append(data)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        dataset = PubMedQADataset.load_single(path)
        return dataset
