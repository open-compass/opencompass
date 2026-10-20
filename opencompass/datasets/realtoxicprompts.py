from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class RealToxicPromptsDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        challenging_subset = kwargs.pop('challenging_subset', False)
        if kwargs['path'] == 'allenai/real-toxicity-prompts':
            try:
                dataset = load_dataset(**kwargs)
            except ConnectionError as e:
                raise ConnectionError(
                    f'{e} Something wrong with this dataset, '
                    'cannot track it online or use offline mode, '
                    'please set local file path directly.')
        else:
            path = kwargs.pop('path')
            path = get_data_path(path, local_mode=True)
            dataset = Dataset.from_file(path)
            dataset = DatasetDict(train=dataset)

        def preprocess(example):

            for k, v in example['prompt'].items():
                k = 'prompt_' + k
                example[k] = v
            del example['prompt']

            return example

        dataset = dataset.map(preprocess)

        # return challenging subset if necessary
        if challenging_subset:
            return dataset.filter(lambda example: example['challenging'])
        return dataset
