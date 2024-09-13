from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class LongBenchmusiqueDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        dataset = load_dataset(path=path,
                               name=name,
                               data_dir=path,
                               trust_remote_code=True)
        split = 'test'
        raw_data = []
        for i in range(len(dataset[split])):
            question = dataset[split]['input'][i]
            context = dataset[split]['context'][i]
            answers = dataset[split]['answers'][i]
            raw_data.append({
                'input': question,
                'context': context,
                'answers': answers
            })
        dataset[split] = Dataset.from_list(raw_data)
        return dataset
