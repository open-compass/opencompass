import json
from os import environ

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class XsumDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path, split='validation')
            rows = []
            for i, line in enumerate(ms_dataset):
                if i == 1000:
                    break
                dialogue = line['document']
                summary = line['summary']
                if not dialogue or not summary:
                    continue
                rows.append({'dialogue': dialogue, 'summary': summary})
            dataset = Dataset.from_list(rows)
        else:
            with open(path, 'r', errors='ignore') as in_f:
                rows = []
                for i, line in enumerate(in_f):
                    if i == 1000:
                        break
                    sample = json.loads(line.strip())
                    dialogue = sample['dialogue']
                    summary = sample['summary']
                    if isinstance(dialogue, float) or isinstance(
                            summary, float):
                        continue
                    rows.append({'dialogue': dialogue, 'summary': summary})
                dataset = Dataset.from_dict({
                    'dialogue': [row['dialogue'] for row in rows],
                    'summary': [row['summary'] for row in rows]
                })
        return dataset


@TEXT_POSTPROCESSORS.register_module('Xsum')
def Xsum_postprocess(text: str) -> str:
    text = text.strip().split('\n')[0].strip()
    return text
