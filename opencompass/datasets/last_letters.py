import os
import re
import json
from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset
from opencompass.utils.datasets import DEFAULT_DATA_FOLDER
from opencompass.utils.fileio import download_url

@LOAD_DATASET.register_module()
class LastLettersDataset(BaseDataset):
    
    @staticmethod
    def load(path: str):
        cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
        local_path = './data/last_letters/last_letters.json'
        data_path = os.path.join(DEFAULT_DATA_FOLDER, cache_dir, local_path)
        
        if not os.path.exists(data_path):
            dataset_url = "https://raw.githubusercontent.com/wjn1996/Chain-of-Knowledge/refs/heads/main/tasks/Letter/dataset/last_letters.json"
            download_url(dataset_url, os.path.dirname(data_path))
            
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for ex in json.load(f)["examples"]:
                dataset.append(ex)
        dataset = Dataset.from_list(dataset)
        return DatasetDict({'test': dataset})
    
@TEXT_POSTPROCESSORS.register_module('last_letters')
def last_letters_pred_postprocess(text: str) -> str:
    text = text.split('answer is ')[-1]
    text = re.sub("\"|\'|\n|\.|\s", "", text)
    text = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", text)
    return text