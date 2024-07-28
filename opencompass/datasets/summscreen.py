from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SummScreenDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        import json
        import os
        dataset_dict = DatasetDict()
        split = 'dev'
        dev_list = []
        fd_folder = os.path.join(path, 'SummScreen_raw', 'fd')
        files = os.listdir(fd_folder)
        for file in files:
            filename = os.path.join(fd_folder, file)
            with open(filename, 'r') as f:
                data = json.load(f)
                summary = ''.join(data['Recap'])
                content = '\n'.join(data['Transcript'])
                dev_list.append({
                    'content': content,
                    'summary': summary,
                })

        tms_folder = os.path.join(path, 'SummScreen_raw', 'tms')
        files = os.listdir(tms_folder)
        for file in files:
            filename = os.path.join(tms_folder, file)
            with open(filename, 'r') as f:
                data = json.load(f)
                summary = ''.join(data['Recap'])
                content = '\n'.join(data['Transcript'])
                dev_list.append({
                    'content': content,
                    'summary': summary,
                })
        dataset_dict[split] = Dataset.from_list(dev_list)
        return dataset_dict
