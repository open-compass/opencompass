from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class LEvalTPODataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)
        split = 'test'
        raw_data = []
        for i in range(len(dataset[split])):
            instructions = dataset[split]['instructions'][i]
            outputs = dataset[split]['outputs'][i]
            context = dataset[split]['input'][i]
            for question, answer in zip(instructions, outputs):
                raw_data.append({
                    'question': question,
                    'context': context,
                    'answer': answer
                })
        dataset[split] = Dataset.from_list(raw_data)
        return dataset
