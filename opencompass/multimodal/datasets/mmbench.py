import base64
import io
from typing import List, Optional

import pandas as pd
from mmengine.dataset import Compose
from PIL import Image
from torch.utils.data import Dataset

from opencompass.registry import DATASETS


def decode_base64_to_image(base64_string) -> Image:
    """Convert raw data into Pillow image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


@DATASETS.register_module()
class MMBenchDataset(Dataset):
    """Dataset to load MMBench dataset.

    Args:
        data_file (str): The path of the dataset.
        pipeline (dict): The data augmentation.
        sys_prompt (str): The system prompt added to the head
            of these options. Defaults to
            There are several options:
    """

    def __init__(self,
                 data_file: str,
                 pipeline: List[dict],
                 sys_prompt: str = 'There are several options:') -> None:
        self.df = pd.read_csv(data_file, sep='\t')
        self.pipeline = Compose(pipeline)
        self.sys_prompt = sys_prompt

    def __len__(self) -> None:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = decode_base64_to_image(image)
        question = self.df.iloc[idx]['question']
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        hint = self.load_from_df(idx, 'hint')
        data = {
            'img': image,
            'question': question,
            'options': options_prompt,
            'category': catetory,
            'l2-category': l2_catetory,
            'options_dict': options,
            'index': index,
            'context': hint,
        }
        data = self.pipeline(data)
        return data

    def load_from_df(self, idx: int, key: str) -> Optional[str]:
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None
