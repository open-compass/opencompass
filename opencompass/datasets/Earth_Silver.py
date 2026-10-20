from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class Earth_Silver_MCQDataset(BaseDataset):

    @staticmethod
    def load(path: str, prompt_mode: str = 'zero-shot', **kwargs):

        dataset = load_dataset(path=path, split='multiple_choice')

        dataset = dataset.map(lambda item: {
            'question': item['question'],
            'answer': item['answer']
        })

        if prompt_mode == 'zero-shot':
            return dataset
        elif prompt_mode == 'few-shot':
            raise NotImplementedError('few-shot prompt 尚未实现')
        else:
            raise ValueError(f'Unsupported prompt_mode: {prompt_mode}')
