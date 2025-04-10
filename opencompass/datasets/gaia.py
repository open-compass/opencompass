import json
from os import environ
import os

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class GAIADataset(BaseDataset):

    @staticmethod
    def load(path, local_mode: bool = False):
        
        from datasets import load_dataset
        try:
            # 因为ModelScope的GAIA数据集读取存在问题，所以从huggingface读取
            ds = load_dataset("gaia-benchmark/GAIA", '2023_all', split='validation')
            rows = []
            for item in ds:
                rows.append({
                    'question': item['Question'],
                    'answerKey': item['Final answer'],
                    'file_path': item['file_path'],
                    'file_name': item['file_name'],
                    'level': item['Level']
                })
        except Exception as e:
            print(f"Error loading local file: {e}")
            
        return Dataset.from_list(rows)
