import json
import os

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class commonsenseqaDataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = {}
        for split, stub in [
            ['train', 'train_rand_split.jsonl'],
            ['validation', 'dev_rand_split.jsonl'],
        ]:
            data_path = os.path.join(path, stub)
            dataset_list = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    dataset_list.append({
                        'question':
                        line['question']['stem'],
                        'A':
                        line['question']['choices'][0]['text'],
                        'B':
                        line['question']['choices'][1]['text'],
                        'C':
                        line['question']['choices'][2]['text'],
                        'D':
                        line['question']['choices'][3]['text'],
                        'E':
                        line['question']['choices'][4]['text'],
                        'answerKey':
                        line['answerKey'],
                    })
            dataset[split] = Dataset.from_list(dataset_list)

        return DatasetDict(dataset)
