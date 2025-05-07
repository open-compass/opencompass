from datasets import load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ClinicBenchDataset(BaseDataset):

    @staticmethod
    def load_single(path):
        dataset = load_dataset(path)['train']
        return dataset

    @staticmethod
    def load(path):
        path = get_data_path(path)
        dataset = ClinicBenchDataset.load_single(path)
        return dataset
