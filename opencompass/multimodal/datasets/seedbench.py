import os.path as osp
import json
from typing import List
from PIL import Image

from mmengine.dataset import Compose
from torch.utils.data import Dataset

from opencompass.registry import DATASETS


@DATASETS.register_module()
class SEEDBenchDataset(Dataset):
    """Dataset to load SEED-Bench dataset.

    Args:
        ann_file (str): The path of the annotation file.
        cc3m_path (str): The data path of the image dimension(1-9).
        sthv2_path (str): The data path of the dimention 10.
        epic_kitchens_path (str): The data path of the dimention 11.
        breakfast_path (str): The data path of the dimention 12.
        image_pipeline (List[dict]): The data transforms for image.
        video_pipeline (List[dict]): The data transforms for video.
    """

    def __init__(
        self,
        ann_file: str,
        cc3m_path: str,
        sthv2_path: str,
        epic_kitchens_path: str,
        breakfast_path: str,
        image_pipeline: List[dict],
        video_pipeline: List[dict],
    ) -> None:
        ann_file = json.load(open(ann_file, 'rb'))
        if 'questions' in ann_file.keys():
            self.ann_file = ann_file['questions']
        self.cc3m_path = cc3m_path
        self.sthv2_path = sthv2_path
        self.epic_kitchens_path = epic_kitchens_path
        self.breakfast_path = breakfast_path
        self.image_pipeline = Compose(image_pipeline)
        # self.video_pipeline = Compose(video_pipeline)

    def __len__(self) -> None:
        return len(self.ann_file)

    def __getitem__(self, idx: str) -> dict:
        item = self.ann_file[idx]
        data = {
            'question':
            item['question'],
            'answer':
            item['answer'],
            'choices': [
                item['choice_a'], item['choice_b'], item['choice_c'],
                item['choice_d']
            ],
            'data_type':
            item['data_type'],
            'question_id':
            item['question_id'],
            'question_type_id':
            item['question_type_id'],
            'index':
            idx,
        }

        if item['data_type'] == 'image':
            data_path = osp.join(self.cc3m_path, item['data_id'])
            raw_image = Image.open(open(data_path, "rb")).convert("RGB")
            data['data_path'] = data_path
            data['img'] = raw_image
            data = self.image_pipeline(data)
        elif item['data_type'] == 'video':
            return None
            if item['question_type_id'] == 10:
                data_path = osp.join(self.sthv2_path, item['data_id'])
                data['data_path'] = data_path
            elif item['question_type_id'] == 11:
                data_path = osp.join(self.epic_kitchens_path, item['data_id'])
                data['data_path'] = data_path
                data['segment'] = item['segment']
            elif item['question_type_id'] == 12:
                data_path = osp.join(self.breakfast_path, item['data_id'])
                data['data_path'] = data_path
                data['segment'] = item['segment']
            else:
                raise ValueError("The question type id is not valid.")

            data = self.video_pipeline(data)

        else:
            raise ValueError("The data type is not valid.")

        return data
