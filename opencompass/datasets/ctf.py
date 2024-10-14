import datasets
from .base import BaseDataset

class MyDataset(BaseDataset):

    @staticmethod
    def load(**kwargs) -> datasets.Dataset:
        pass