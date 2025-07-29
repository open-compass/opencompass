import os.path as osp
from os import environ
from ..base import BaseDataset
from .utils import load_jsonl_data
from opencompass.utils import get_data_path
from opencompass.registry import LOAD_DATASET

@LOAD_DATASET.register_module()
class EESEDataset(BaseDataset):

    @staticmethod
    def load(path: str, file_name: str = 'EESE.jsonl', **kwargs):
        """Load EESE dataset from HuggingFace or local file.
        
        Args:
            path (str): Path to the dataset
            file_name (str): Name of the JSONL file
            **kwargs: Additional arguments
            
        Returns:
            datasets.Dataset: The loaded EESE dataset
        """
        path = get_data_path(path)
        
        if environ.get('DATASET_SOURCE') == 'HuggingFace':
            from datasets import load_dataset
            dataset = load_dataset("AIBench/EESE", "default", split="test")
        else:
            from datasets import Dataset, DatasetDict
            dataset = DatasetDict()
            for split in ['test']:
                raw_data = []
                filename = osp.join(path, split, f'{file_name}')
                raw_data = load_jsonl_data(filename)
            dataset['test'] = Dataset.from_list(raw_data)
            dataset['train'] = Dataset.from_list(raw_data)
        
        return dataset
