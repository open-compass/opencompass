import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class hellaswagDataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    'ctx': data['query'].split(': ', 2)[-1],
                    'A': data['choices'][0],
                    'B': data['choices'][1],
                    'C': data['choices'][2],
                    'D': data['choices'][3],
                    'label': data['gold'],
                })
        dataset = Dataset.from_list(dataset)
        return dataset


@LOAD_DATASET.register_module()
class hellaswagDataset_V2(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    'ctx': data['query'].split(': ', 1)[-1],
                    'A': data['choices'][0],
                    'B': data['choices'][1],
                    'C': data['choices'][2],
                    'D': data['choices'][3],
                    'label': 'ABCD'[data['gold']],
                })
        dataset = Dataset.from_list(dataset)
        return dataset


@LOAD_DATASET.register_module()
class hellaswagDataset_V3(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    'query': data['query'],
                    'A': data['choices'][0],
                    'B': data['choices'][1],
                    'C': data['choices'][2],
                    'D': data['choices'][3],
                    'gold': data['gold'],
                })
        dataset = Dataset.from_list(dataset)
        return dataset


@LOAD_DATASET.register_module()
class hellaswagDatasetwithICE(BaseDataset):

    @staticmethod
    def load(path):
        dataset_dict = DatasetDict()
        for split, filename in [
            ['train', 'hellaswag_train_sampled25.jsonl'],
            ['val', 'hellaswag.jsonl'],
        ]:
            dataset = []
            with open(osp.join(path, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    dataset.append({
                        'ctx': data['query'].split(': ', 1)[-1],
                        'A': data['choices'][0],
                        'B': data['choices'][1],
                        'C': data['choices'][2],
                        'D': data['choices'][3],
                        'label': 'ABCD'[data['gold']],
                    })
            dataset_dict[split] = Dataset.from_list(dataset)
        return dataset_dict


class hellaswagDatasetClean(BaseDataset):

    # load the contamination annotations of CEval from
    # https://github.com/liyucheng09/Contamination_Detector
    @staticmethod
    def load_contamination_annotations(path, split='val'):
        import requests

        assert split == 'val', 'We only use val set of hellaswag'
        annotation_cache_path = osp.join(
            path, f'hellaswag_{split}_contamination_annotations.json')
        if osp.exists(annotation_cache_path):
            with open(annotation_cache_path, 'r') as f:
                annotations = json.load(f)
            return annotations
        link_of_annotations = 'https://github.com/liyucheng09/Contamination_Detector/releases/download/v0.1.1rc2/hellaswag_annotations_with_line_index.json'  # noqa
        annotations = json.loads(requests.get(link_of_annotations).text)
        with open(annotation_cache_path, 'w') as f:
            json.dump(annotations, f)
        return annotations

    @staticmethod
    def load(path):
        dataset = []
        annotations = hellaswagDatasetClean.load_contamination_annotations(
            osp.dirname(path))
        with open(path, 'r', encoding='utf-8') as f:
            for rwo_index, line in enumerate(f):
                data = json.loads(line)
                rwo_index = f'{rwo_index}'
                if rwo_index in annotations:
                    is_clean = annotations[rwo_index][0]
                else:
                    is_clean = 'not labeled'
                dataset.append({
                    'ctx': data['query'].split(': ', 2)[-1],
                    'A': data['choices'][0],
                    'B': data['choices'][1],
                    'C': data['choices'][2],
                    'D': data['choices'][3],
                    'label': data['gold'],
                    'is_clean': is_clean,
                })
        dataset = Dataset.from_list(dataset)
        return dataset
