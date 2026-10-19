import json
import os.path as osp
from typing import Dict, Optional

import mmengine
from datasets import Dataset, DatasetDict

from opencompass.registry import TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from ..base import BaseDataset


class TEvalDataset(BaseDataset):

    def __init__(self, reader_cfg: Optional[Dict] = {}, **kwargs):
        super().__init__(reader_cfg=reader_cfg, **kwargs)

    def load(self, path: str, name: str):
        path = get_data_path(path, local_mode=True)

        dataset = DatasetDict()
        data = mmengine.load(osp.join(path, f'{name}.json'))
        raw_data = []
        for i in data.keys():
            origin_prompt = data[i]['origin_prompt']
            if isinstance(origin_prompt, str):
                origin_prompt = json.loads(origin_prompt)
            # Aligning the default roles of opencompass
            prompt = origin_prompt + [
                dict(role='assistant',
                     content=str(data[i].get('ground_truth')))
            ]
            raw_data.append({
                'prompt': prompt,
                'ground_truth': json.dumps(data[i])
            })
        dataset['test'] = Dataset.from_list(raw_data)
        dataset['train'] = Dataset.from_list(raw_data)
        return dataset




@TEXT_POSTPROCESSORS.register_module('teval')
def teval_postprocess(text: str) -> str:
    if isinstance(text, str):
        text = text.split('<eoa>')[0]
        text = text.split('<TOKENS_UNUSED_1>')[0]
        text = text.split('<|im_end|>')[0]
        text = text.split('\nuser')[0]
        text = text.split('\nUSER')[0]
        text = text.split('[INST]')[0]
        text = text.strip()
        if text.startswith('```json'):
            text = text[len('```json'):]
        text = text.strip('`').strip()
        if text[:2] == '{{' and text[-2:] == '}}':
            text = text[1:-1]
    return str(text)
