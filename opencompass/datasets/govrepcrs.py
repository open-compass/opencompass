import json
import os

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class GovRepcrsDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)

        dataset_dict = DatasetDict()
        splits = ['train', 'valid', 'test']
        dataset_lists = {x: [] for x in splits}
        for split in splits:
            split_fp = os.path.join(path, 'gov-report', 'split_ids',
                                    'crs_' + split + '.ids')
            with open(split_fp, 'r') as f:
                for line in f.readlines():
                    xpath = os.path.join(path, 'gov-report', 'crs',
                                         line.strip() + '.json')
                    with open(xpath, 'r') as df:
                        data = json.load(df)
                        content = data['title'] + '\n' + '\n'.join(
                            [(x['section_title'] if x['section_title'] else '')
                             + '\n' + '\n'.join(x['paragraphs'])
                             for x in data['reports']['subsections']])
                        summary = '\n'.join(data['summary'])
                        dataset_lists[split].append({
                            'content': content,
                            'summary': summary,
                        })
                dataset_dict[split] = Dataset.from_list(dataset_lists[split])
        return dataset_dict
