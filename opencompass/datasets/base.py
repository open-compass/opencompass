from copy import deepcopy
from typing import Dict, List, Optional, Union

from datasets import Dataset, DatasetDict

from opencompass.openicl import DatasetReader


class BaseDataset:

    def __init__(self,
                 reader_cfg: Optional[Dict] = {},
                 k: Union[int, List[int]] = 1,
                 n: int = 1,
                 **kwargs):
        abbr = kwargs.pop('abbr', 'dataset')
        dataset = self.load(**kwargs)
        # maybe duplicate
        assert (max(k) if isinstance(k, List) else
                k) <= n, 'Maximum value of `k` must less than or equal to `n`'
        if isinstance(dataset, Dataset):
            examples = []
            for idx, example in enumerate(dataset):
                if 'subdivision' not in example:
                    example['subdivision'] = abbr
                if 'idx' not in example:
                    example['idx'] = idx
                examples.append(example)
            examples = sum([deepcopy(examples) for _ in range(n)], [])
            self.dataset = Dataset.from_list(examples)
        else:
            self.dataset = DatasetDict()
            for key in dataset:
                examples = []
                for idx, example in enumerate(dataset[key]):
                    if 'subdivision' not in example:
                        example['subdivision'] = f'{abbr}_{key}'
                    if 'idx' not in example:
                        example['idx'] = idx
                    examples.append(example)
                print(abbr, key, len(examples))
                examples = sum([deepcopy(examples) for _ in range(n)], [])
                self.dataset[key] = Dataset.from_list(examples)
        self._init_reader(**reader_cfg)

    def _init_reader(self, **kwargs):
        self.reader = DatasetReader(self.dataset, **kwargs)

    @property
    def train(self):
        return self.reader.dataset['train']

    @property
    def test(self):
        return self.reader.dataset['test']

    @staticmethod
    def load(**kwargs) -> Union[Dataset, DatasetDict]:
        pass
