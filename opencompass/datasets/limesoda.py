import os
import json
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset

@LOAD_DATASET.register_module()
class LimesodaDataset(BaseDataset):
    @staticmethod
    def load(path):
        dataset = DatasetDict()
        for file in os.listdir(path):
            if '.jsonl' not in file: continue
            split = file.split('.')[0]
            origin_data = [json.loads(line.strip()) for line in open(os.path.join(path, file))]
            dataset[split] = Dataset.from_list([{'document': item['input'], 'label': item['target']} for item in origin_data])
        return dataset
    
