from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ScienceQADataset(BaseDataset):

    @staticmethod
    def load_single():
        dataset = []
        ds = load_dataset('derek-thomas/ScienceQA')
        num = 0
        for data in ds['test']:
            if data['image'] is None and data['topic'] == 'biology':
                num += 1
                # if num > 10:
                #     break
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
        train_dataset = Dataset.from_list([])
        val_dataset = ScienceQADataset.load_single()
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        return dataset
