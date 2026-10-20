from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class LVEvalmultifieldqaenDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        if 'data_files' in kwargs:
            kwargs['data_files'] = get_data_path(kwargs['data_files'],
                                                 local_mode=True)
        dataset = load_dataset(**kwargs)
        split = 'test'
        raw_data = []
        for i in range(len(dataset[split])):
            question = dataset[split]['input'][i]
            context = dataset[split]['context'][i]
            answers = dataset[split]['answers'][i]
            confusing_facts = dataset[split]['confusing_facts'][i]
            answer_keywords = dataset[split]['answer_keywords'][i]
            answers_with_ak = answers + [answer_keywords]
            raw_data.append({
                'input': question,
                'context': context,
                'answers': answers_with_ak,
                'confusing_facts': confusing_facts,
                'answer_keywords': answer_keywords,
            })
        dataset[split] = Dataset.from_list(raw_data)
        return dataset
