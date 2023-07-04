from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class RealToxicPromptsDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        challenging_subset = kwargs.pop('challenging_subset', False)
        dataset = load_dataset(**kwargs)

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
