from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ScienceQADataset(BaseDataset):

    @staticmethod
    def load_single(path):
        dataset = []
        ds = load_dataset(path)
        for data in ds['test']:
            if data['image'] is None:
                data['label'] = chr(65 + data['answer']
                                    ) + '. ' + data['choices'][data['answer']]
                choices = ''
                for i in range(len(data['choices'])):
                    choices += chr(65 + i) + '. ' + data['choices'][i] + '\n'
                data['choices'] = choices
                # print(data)

                dataset.append(data)

        return Dataset.from_list(dataset)

    @staticmethod
    def load(path):
        dataset = ScienceQADataset.load_single(path)
        return dataset
