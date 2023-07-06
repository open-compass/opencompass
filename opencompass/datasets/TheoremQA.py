import re

from datasets import load_dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class TheoremQADataset(BaseDataset):

    @staticmethod
    def load(path: str):
        return load_dataset('csv', data_files={'test': path})


@TEXT_POSTPROCESSORS.register_module('TheoremQA')
def TheoremQA_postprocess(text: str) -> str:
    text = text.strip()
    matches = re.findall(r'answer is ([^\s]+)', text)
    if len(matches) == 0:
        return text
    else:
        text = matches[0].strip().strip('.,?!\"\';:')
        return text
