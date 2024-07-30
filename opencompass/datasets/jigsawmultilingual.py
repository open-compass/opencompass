import csv

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class JigsawMultilingualDataset(BaseDataset):

    @staticmethod
    def load(path, label, lang):
        path = get_data_path(path, local_mode=True)
        label = get_data_path(label, local_mode=True)

        assert lang in ['es', 'fr', 'it', 'pt', 'ru', 'tr']
        dataset = DatasetDict()

        data_list = list()
        idx = 0
        with open(path) as file, open(label) as label:
            text_reader = csv.reader(file)
            label_reader = csv.reader(label)
            for text, target in zip(text_reader, label_reader):
                if text[2] == lang:
                    assert text[0] == target[0]
                    data_list.append({
                        'idx': idx,
                        'text': text[1],
                        'label': int(target[1]),
                        'choices': ['no', 'yes']
                    })
                    idx += 1

        dataset['test'] = Dataset.from_list(data_list)
        return dataset
