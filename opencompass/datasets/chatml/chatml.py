import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class ChatMLDataset(BaseDataset):

    @staticmethod
    def load(path, file_name=None, local_mode=False):

        path = get_data_path(path, local_mode=local_mode)
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = [json.loads(line) for line in f]

        from .verification import VerifyDataset
        for i in data:
            VerifyDataset(**i)

        input_prompt = '\nRemember to put your final answer within \\boxed{}.'
        for i in range(len(data)):
            for j in range(len(data[i]['question'])):
                if data[i]['question'][j]['role'] == 'user':
                    data[i]['question'][j]['content'] += input_prompt

        return Dataset.from_list(data)
