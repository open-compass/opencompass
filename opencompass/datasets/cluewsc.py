import json

from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CluewscDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        if 'data_files' in kwargs:
            kwargs['data_files'] = get_data_path(kwargs['data_files'],
                                                 local_mode=True)
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            text_list = list(example['text'])
            # span1 may have 1 or more than 1 words
            # span2 is the pronoun and has only 1 word
            text_list[example['target']
                      ['span2_index']] = example['target']['span1_text']
            example['new_text'] = ''.join(text_list)
            if example['label'] == 'true':
                example['answer'] = 1
            else:
                example['answer'] = 0
            example['span1'] = example['target']['span1_text']
            example['span2'] = example['target']['span2_text']
            del example['target']
            return example

        dataset = dataset.map(preprocess)
        return dataset


@LOAD_DATASET.register_module()
class CluewscDatasetV2(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                item = {
                    'span1': line['target']['span1_text'],
                    'span2': line['target']['span2_text'],
                    'text': line['text'],
                    'label': {
                        'true': 'A',
                        'false': 'B'
                    }[line['label']],
                }
                data.append(item)
        return Dataset.from_list(data)
