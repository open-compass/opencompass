import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

# HuggingFace Hub repository hosting the dataset
HF_REPO_ID = 'cindy90/BuySideFinBench'


@LOAD_DATASET.register_module()
class BuySideFinBenchDataset(BaseDataset):
    """BuySideFinBench: bilingual buy-side equity research benchmark.

    Loading priority:
      1. Local CSV files at ``path/{dev,test}/{name}.csv`` if available
      2. Auto-download from HuggingFace Hub (``cindy90/BuySideFinBench``)

    Local CSV format: 7 columns
        [id, question, A, B, C, D, answer]
    """

    @staticmethod
    def load(path: str, name: str):
        resolved_path = get_data_path(path, local_mode=False)

        # Prefer local CSV when present
        if resolved_path and osp.isdir(resolved_path):
            local_test_file = osp.join(resolved_path, 'test', f'{name}.csv')
            if osp.isfile(local_test_file):
                return BuySideFinBenchDataset._load_from_local(
                    resolved_path, name)

        # Fall back to HuggingFace Hub auto-download
        return BuySideFinBenchDataset._load_from_hf(name)

    @staticmethod
    def _load_from_local(path: str, name: str) -> DatasetDict:
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                _ = next(reader)  # skip header
                for row in reader:
                    assert len(row) == 7, (
                        f'Expected 7 columns in {filename}, got {len(row)}')
                    raw_data.append({
                        'question': row[1],
                        'A': row[2],
                        'B': row[3],
                        'C': row[4],
                        'D': row[5],
                        'answer': row[6],
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset

    @staticmethod
    def _load_from_hf(name: str) -> DatasetDict:
        from datasets import load_dataset

        hf_ds = load_dataset(HF_REPO_ID, name=name)

        dataset = DatasetDict()
        for split in ['dev', 'test']:
            if split not in hf_ds:
                raise ValueError(
                    f'Split "{split}" not found in {HF_REPO_ID}/{name}. '
                    f'Available: {list(hf_ds.keys())}')
            dataset[split] = hf_ds[split]
        return dataset
