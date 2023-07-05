from datasets import DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class dropDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs, split='validation')

        def pre_process(example):
            example['answers'] = example['answers_spans']['spans']
            example['prompt'] = example.pop('passage')
            return example

        def only_number(example):
            for i in example['answers_spans']['types']:
                if i == 'number':
                    return True
            return False

        dataset = dataset.filter(only_number)
        dataset = dataset.map(pre_process).remove_columns(
            ['section_id', 'query_id'])
        return DatasetDict({'validation': dataset})
