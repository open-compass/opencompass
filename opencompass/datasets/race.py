from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class RaceDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = load_dataset(path, name)

        def preprocess(x):
            for ans, option in zip(['A', 'B', 'C', 'D'], x['options']):
                x[ans] = option
            del x['options']
            return x

        return dataset.map(preprocess)
