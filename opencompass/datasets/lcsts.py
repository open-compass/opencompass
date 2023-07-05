import os.path as osp

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class LCSTSDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        src_path = osp.join(path, 'test.src.txt')
        tgt_path = osp.join(path, 'test.tgt.txt')

        src_lines = open(src_path, 'r', encoding='utf-8').readlines()
        tgt_lines = open(tgt_path, 'r', encoding='utf-8').readlines()

        data = {'content': [], 'abst': []}

        for _, (src_text, tgt_text) in enumerate(zip(src_lines, tgt_lines)):
            data['content'].append(src_text.strip())
            data['abst'].append(tgt_text.strip())

        dataset = Dataset.from_dict({
            'content': data['content'],
            'abst': data['abst']
        })
        return dataset


@TEXT_POSTPROCESSORS.register_module('lcsts')
def lcsts_postprocess(text: str) -> str:
    text = text.strip().split('\n')[0].replace('своей', '').strip()
    text = text.replace('1. ', '') if text.startswith('1. ') else text
    text = text.replace('- ', '') if text.startswith('- ') else text
    text = text.strip('“，。！”')
    return text
