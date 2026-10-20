import re

from datasets import load_dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class TheoremQADataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)        
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


def TheoremQA_postprocess_v2(text: str) -> str:
    prediction = text.strip().strip('\n').split('\n')[-1]
    tmp = ''
    for entry in prediction.split(' ')[::-1]:
        if entry == 'is' or entry == 'be' or entry == 'are' or entry.endswith(
                ':'):
            break
        tmp = entry + ' ' + tmp
    prediction = tmp.strip().strip('.')
    return prediction
