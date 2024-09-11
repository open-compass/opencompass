from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class LongBenchtrecDataset(BaseDataset):

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


@TEXT_POSTPROCESSORS.register_module()
def trec_postprocess(text: str) -> str:
    text = text.lstrip('\n').split('\n')[0]
    return text
