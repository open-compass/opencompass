import pandas as pd

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CoLADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):

        train_df = pd.read_csv(f'{path}/in_domain_train.tsv', 
                               sep='\t', 
                               usecols=[1, 3], 
                               index_col=None,
                               names=['label', 'sentence'])
        dev_df = pd.read_csv(f'{path}/{name}_dev.tsv', 
                             sep='\t', 
                             usecols=[1, 3], 
                             index_col=None, 
                             names=['label', 'sentence'])

        return DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'dev': Dataset.from_pandas(dev_df)
        })
