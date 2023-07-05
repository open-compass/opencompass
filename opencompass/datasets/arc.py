import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ARCDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        with open(path, 'r', errors='ignore') as in_f:
            rows = []
            for i, line in enumerate(in_f):
                sample = json.loads(line.strip())
                answerKey = sample['answerKey']
                sample = sample['question']
                question = sample['stem']
                choices = sample['choices']
                if len(choices) != 4:
                    continue
                textA = choices[0]['text']
                textB = choices[1]['text']
                textC = choices[2]['text']
                textD = choices[3]['text']
                rows.append({
                    'question': question,
                    'answerKey': answerKey,
                    'textA': textA,
                    'textB': textB,
                    'textC': textC,
                    'textD': textD
                })
            dataset = Dataset.from_dict({
                'question': [row['question'] for row in rows],
                'answerKey': [row['answerKey'] for row in rows],
                'textA': [row['textA'] for row in rows],
                'textB': [row['textB'] for row in rows],
                'textC': [row['textC'] for row in rows],
                'textD': [row['textD'] for row in rows]
            })
            return dataset
