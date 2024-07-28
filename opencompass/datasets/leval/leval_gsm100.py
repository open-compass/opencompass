from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from ..base import BaseDataset


@TEXT_POSTPROCESSORS.register_module('gsm100_dataset')
def gsm100_dataset_postprocess(text: str) -> str:
    return text.replace(',', '')


@TEXT_POSTPROCESSORS.register_module('gsm100')
def gsm100_postprocess(text: str) -> str:
    # text = text.split('\n\n')[0]
    segs = text.split('The answer is')
    if len(segs) < 2:
        return ''
    text = segs[1]
    text = text.split(' ')
    flag = False
    ret = ''
    for i in range(len(text)):
        s = text[i]
        for i in range(len(s)):
            if s[i].isdigit():
                flag = True
                ret = s
                break
        if flag:
            break
    ret1 = ''
    for i in range(len(ret)):
        if ret[i].isdigit():
            ret1 += ret[i]
    return ret1


@LOAD_DATASET.register_module()
class LEvalGSM100Dataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        if 'data_files' in kwargs:
            kwargs['data_files'] = get_data_path(kwargs['data_files'],
                                                 local_mode=True)
        dataset = load_dataset(**kwargs)
        split = 'test'
        raw_data = []
        for i in range(len(dataset[split])):
            instructions = dataset[split]['instructions'][i]
            outputs = dataset[split]['outputs'][i]
            context = dataset[split]['input'][i]
            for question, answer in zip(instructions, outputs):
                raw_data.append({
                    'question': question,
                    'context': context,
                    'answer': answer
                })
        dataset[split] = Dataset.from_list(raw_data)
        return dataset
