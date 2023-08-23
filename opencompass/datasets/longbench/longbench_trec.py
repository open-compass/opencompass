from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class LongBenchtrecDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)
        split = 'test'
        raw_data = []
        for i in range(len(dataset[split])):
            question = dataset[split]['input'][i]
            context = dataset[split]['context'][i]
            answers = dataset[split]['answers'][i]
            all_classes = dataset[split]['all_classes'][i]
            raw_data.append({
                'input': question,
                'context': context,
                'all_labels': {
                    'answers': answers,
                    'all_classes': all_classes
                }
            })
        dataset[split] = Dataset.from_list(raw_data)
        return dataset
