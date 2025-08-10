import json
import os.path as osp
import re

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ScibenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path, local_mode=True)
        train_data = []

        filename = osp.join(path, f'{name}.json')
        with open(filename, 'r') as infile:
            raw_data = json.load(infile)

        for entry in raw_data:
            train_data.append({
                'question': entry['problem_text'].strip(),
                'answer': entry['answer_number'].strip()
            })

        dataset = Dataset.from_list(train_data)
        return dataset


@TEXT_POSTPROCESSORS.register_module('scibench')
def scibench_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()

    match = re.search(r'\\boxed\{(.+?)\}', ans)
    if match:
        extracted_content = match.group(1)
        return extracted_content

    output = re.sub(r'(\d),(\d)', r'\1\2', ans)
    numbers = re.findall(r'-?\d*\.?\d+|\d+', output)
    if numbers:
        return numbers[-1]

    return ans
