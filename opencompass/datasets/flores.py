import os
import re

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class FloresFirst100Dataset(BaseDataset):

    @staticmethod
    def load_single(src_path, tgt_path, src_lang, tgt_lang):

        with open(src_path, 'r', encoding='utf-8') as f:
            src_lines = f.readlines()
        with open(tgt_path, 'r', encoding='utf-8') as f:
            tgt_lines = f.readlines()
        assert len(src_lines) == len(tgt_lines)
        dataset_list = [{
            f'sentence_{src_lang}': src_lines[i].strip(),
            f'sentence_{tgt_lang}': tgt_lines[i].strip(),
        } for i in range(len(src_lines))]
        return Dataset.from_list(dataset_list)

    @staticmethod
    def load(path, name):
        src_lang, tgt_lang = name.split('-')
        dev_dataset = FloresFirst100Dataset.load_single(
            os.path.join(path, 'dev', f'{src_lang}.dev'),
            os.path.join(path, 'dev', f'{tgt_lang}.dev'), src_lang, tgt_lang)
        devtest_dataset = FloresFirst100Dataset.load_single(
            os.path.join(path, 'devtest', f'{src_lang}.devtest'),
            os.path.join(path, 'devtest', f'{tgt_lang}.devtest'), src_lang,
            tgt_lang)
        return DatasetDict({'dev': dev_dataset, 'devtest': devtest_dataset})


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
