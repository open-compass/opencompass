from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MedQADataset(BaseDataset):

    @staticmethod
    def load_single(path):
        dataset = []
        data_lines = load_dataset(path, 'test')  # "data/MedQA"
        num = 0
        for data in data_lines:
            num += 1
            choices = ''
            for i in range(4):
                data[chr(65 + i)] = data['ending' + str(i)]
                choices += chr(65 + i) + '. ' + data['ending' + str(i)] + '\n'
            data['question'] = data['sent1']
            data['choices'] = choices
            data['label'] = chr(65 + int(data['label'])) + '. ' + data[
                'ending' + str(data['label'])]

            dataset.append(data)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        train_dataset = Dataset.from_list([])
        val_dataset = MedQADataset.load_single(path)  # "data/MedQA/test.json"
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        return dataset
