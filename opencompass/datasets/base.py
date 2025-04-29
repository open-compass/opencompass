from typing import Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, concatenate_datasets

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
            dataset = dataset.map(lambda x, idx: {
                'subdivision': abbr,
                'idx': idx
            },
                                  with_indices=True,
                                  writer_batch_size=16)
            dataset = concatenate_datasets([dataset] * n)
            self.dataset = dataset
        else:
            self.dataset = DatasetDict()
            for key in dataset:
                dataset[key] = dataset[key].map(lambda x, idx: {
                    'subdivision': f'{abbr}_{key}',
                    'idx': idx
                },
                                                with_indices=True,
                                                writer_batch_size=16)
                dataset[key] = concatenate_datasets([dataset[key]] * n)
                self.dataset[key] = dataset[key]
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
