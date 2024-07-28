import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MMLUDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        dataset = DatasetDict()
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            for split in ['dev', 'test']:
                # 从 ModelScope 加载数据
                ms_dataset = MsDataset.load(path,
                                            subset_name=name,
                                            split=split)
                dataset_list = []
                for line in ms_dataset:
                    dataset_list.append({
                        'input': line['question'],
                        'A': line['choices'][0],
                        'B': line['choices'][1],
                        'C': line['choices'][2],
                        'D': line['choices'][3],
                        'target': 'ABCD'[line['answer']],
                    })
                dataset[split] = Dataset.from_list(dataset_list)
        else:
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
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope.utils.config_ds import MS_DATASETS_CACHE
            annotation_cache_path = osp.join(
                MS_DATASETS_CACHE,
                f'MMLU_{split}_contamination_annotations.json')
            link_of_annotations = 'https://modelscope.cn/datasets/opencompass/Contamination_Detector/resolve/master/mmlu_annotations.json'  # noqa
        else:
            annotation_cache_path = osp.join(
                path, split, f'MMLU_{split}_contamination_annotations.json')
            link_of_annotations = 'https://github.com/liyucheng09/Contamination_Detector/releases/download/v0.1.1rc2/mmlu_annotations.json'  # noqa

        if osp.exists(annotation_cache_path):
            with open(annotation_cache_path, 'r') as f:
                annotations = json.load(f)
            return annotations

        annotations = json.loads(requests.get(link_of_annotations).text)
        with open(annotation_cache_path, 'w') as f:
            json.dump(annotations, f)
        return annotations

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        dataset = DatasetDict()
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            for split in ['dev', 'test']:
                from modelscope import MsDataset

                # 从 ModelScope 加载数据
                ms_dataset = MsDataset.load(path,
                                            subset_name=name,
                                            split=split)
                if split == 'test':
                    annotations = \
                        MMLUDatasetClean.load_contamination_annotations(
                            path, split)
                dataset_list = []
                for row_index, line in enumerate(ms_dataset):
                    item = {
                        'input': line['question'],
                        'A': line['choices'][0],
                        'B': line['choices'][1],
                        'C': line['choices'][2],
                        'D': line['choices'][3],
                        'target': 'ABCD'[line['answer']],
                    }
                    if split == 'test':
                        row_id = f'{name} {row_index}'
                        if row_id in annotations:
                            is_clean = annotations[row_id][0]
                        else:
                            is_clean = 'not labeled'
                        item['is_clean'] = is_clean
                    dataset_list.append(item)
                dataset[split] = Dataset.from_list(dataset_list)
        else:
            for split in ['dev', 'test']:
                raw_data = []
                filename = osp.join(path, split, f'{name}_{split}.csv')
                if split == 'test':
                    annotations = \
                        MMLUDatasetClean.load_contamination_annotations(
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
