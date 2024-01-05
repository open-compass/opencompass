import csv
import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MMLUDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    assert len(row) == 6
                    raw_data.append({
                        'input': row[0],
                        'A': row[1],
                        'B': row[2],
                        'C': row[3],
                        'D': row[4],
                        'target': row[5],
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset


class MMLUDatasetClean(BaseDataset):

    # load the contamination annotations of CEval from
    # https://github.com/liyucheng09/Contamination_Detector
    @staticmethod
    def load_contamination_annotations(path, split='val'):
        import requests

        assert split == 'test', 'We only use test set for MMLU'
        annotation_cache_path = osp.join(
            path, split, f'MMLU_{split}_contamination_annotations.json')
        if osp.exists(annotation_cache_path):
            with open(annotation_cache_path, 'r') as f:
                annotations = json.load(f)
            return annotations
        link_of_annotations = 'https://github.com/liyucheng09/Contamination_Detector/releases/download/v0.1.1rc2/mmlu_annotations.json'  # noqa
        annotations = json.loads(requests.get(link_of_annotations).text)
        with open(annotation_cache_path, 'w') as f:
            json.dump(annotations, f)
        return annotations

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.csv')
            if split == 'test':
                annotations = MMLUDatasetClean.load_contamination_annotations(
                    path, split)
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row_index, row in enumerate(reader):
                    assert len(row) == 6
                    item = {
                        'input': row[0],
                        'A': row[1],
                        'B': row[2],
                        'C': row[3],
                        'D': row[4],
                        'target': row[5],
                    }
                    if split == 'test':
                        row_id = f'{name} {row_index}'
                        if row_id in annotations:
                            is_clean = annotations[row_id][0]
                        else:
                            is_clean = 'not labeled'
                        item['is_clean'] = is_clean
                    raw_data.append(item)
            dataset[split] = Dataset.from_list(raw_data)
        return dataset
