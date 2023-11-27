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
            for line in in_f:
                item = json.loads(line.strip())
                question = item['question']
                if len(question['choices']) != 4:
                    continue
                labels = [c['label'] for c in question['choices']]
                answerKey = 'ABCD'[labels.index(item['answerKey'])]
                rows.append({
                    'question': question['stem'],
                    'answerKey': answerKey,
                    'textA': question['choices'][0]['text'],
                    'textB': question['choices'][1]['text'],
                    'textC': question['choices'][2]['text'],
                    'textD': question['choices'][3]['text'],
                })
            return Dataset.from_list(rows)
