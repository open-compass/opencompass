import copy

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class EthicsUtilitarianismDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = load_dataset(path=path, name=name)
        new_dataset = DatasetDict()
        splits = ['train', 'validation', 'test']

        for split in splits:
            examples = []
            for example in dataset[split]:
                example1 = copy.deepcopy(example)
                example1['scenario_A'], example1['scenario_B'] = \
                    example['baseline'], example['less_pleasant']
                example1['label'] = 1
                examples.append(example1)
                example2 = copy.deepcopy(example)
                example2['scenario_A'], example2['scenario_B'] = \
                    example['less_pleasant'], example['baseline']
                example2['label'] = 0
                examples.append(example2)
            new_dataset[split] = Dataset.from_list(examples).remove_columns(
                ['baseline', 'less_pleasant'])

        return new_dataset
