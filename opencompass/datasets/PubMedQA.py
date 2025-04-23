import json

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class PubMedQADataset(BaseDataset):

    @staticmethod
    def load_single(file_path):
        dataset = []
        with open(file_path, 'r') as file:
            data_lines = json.load(file)
        num = 0
        for name in data_lines:
            data = data_lines[name]
            num += 1
            # if num > 10:
            #     break
            data['question'] = (f"CONTEXTS: {data['CONTEXTS']}\n"
                                f"QUESTION: {data['QUESTION']}")
            choices = 'A. yes\nB. no\nC. maybe'
            data['choices'] = choices
            if data['final_decision'] == 'yes':
                data['label'] = 'A. yes'
            elif data['final_decision'] == 'no':
                data['label'] = 'B. no'
            else:
                data['label'] = 'C. maybe'
            # print(data)

            dataset.append(data)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        train_dataset = Dataset.from_list([])
        val_dataset = PubMedQADataset.load_single(
            '/fs-computility/ai4sData/shared/'
            'lifescience/benchmark/raw/PubMedQA/ori_pqal.json')
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        return dataset
