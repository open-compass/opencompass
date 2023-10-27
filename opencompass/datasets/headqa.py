from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class HeadQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = load_dataset(path=path, name=name)

        def preprocess(example):
            answers = example.pop('answers')
            choices_str = ''
            for ans in answers:
                choices_str += f"{ans['aid']}. {ans['atext']}\n"
            example['choices'] = choices_str
            return example

        dataset = dataset.map(preprocess).remove_columns(['image'])
        return dataset
