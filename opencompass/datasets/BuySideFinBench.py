from datasets import DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

# HuggingFace Hub repository hosting the dataset
HF_REPO_ID = 'cindy90/BuySideFinBench'


@LOAD_DATASET.register_module()
class BuySideFinBenchDataset(BaseDataset):
    """BuySideFinBench: bilingual buy-side equity research benchmark.

    Dataset is loaded directly from HuggingFace Hub.
    """

    @staticmethod
    def load(path: str, name: str) -> DatasetDict:
        repo_id = path or HF_REPO_ID
        hf_ds = load_dataset(repo_id, name=name)

        dataset = DatasetDict()
        for split in ['dev', 'test']:
            if split not in hf_ds:
                raise ValueError(
                    f'Missing split "{split}" for {repo_id}/{name}. '
                    f'Available: {list(hf_ds.keys())}')
            dataset[split] = hf_ds[split]
        return dataset
