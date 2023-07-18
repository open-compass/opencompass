import re

from datasets import DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class FloresFirst100Dataset(BaseDataset):

    @staticmethod
    def load(name):
        return DatasetDict({
            'dev':
            load_dataset(path='facebook/flores', name=name, split='dev'),
            'devtest':
            load_dataset(path='facebook/flores',
                         name=name,
                         split='devtest[:100]')
        })


@TEXT_POSTPROCESSORS.register_module('flores')
def flores_postprocess(text: str) -> str:
    text = text.strip().split('\n')[0]
    return text


@TEXT_POSTPROCESSORS.register_module('flores-chinese')
def flores_postprocess_chinese(text: str) -> str:
    import jieba
    truncated_text = text.strip().split('\n')[0]
    cleaned_text = re.sub(r'\s+', ' ', truncated_text).strip()
    cleaned_text = ' '.join(jieba.cut(cleaned_text))
    return cleaned_text
