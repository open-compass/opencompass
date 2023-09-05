import os
from typing import List

from mmengine.dataset import Compose
from torch.utils.data import Dataset

from opencompass.registry import DATASETS


@DATASETS.register_module()
class MMEDataset(Dataset):
    """Dataset to load MME dataset.

    Args:
        data_dir (str): The path of the dataset.
        pipeline (List[dict]): The data augmentation.
    """
    tasks = [
        'artwork', 'celebrity', 'code_reasoning', 'color',
        'commonsense_reasoning', 'count', 'existence', 'landmark',
        'numerical_calculation', 'OCR', 'position', 'posters', 'scene',
        'text_translation'
    ]
    sub_dir_name = ('images', 'questions_answers_YN')

    def __init__(self, data_dir: str, pipeline: List[dict]) -> None:
        self.pipeline = Compose(pipeline)
        self.load_data(data_dir)

    def load_data(self, data_dir: str):
        self.data_list = []
        image_dir, question_dir = self.sub_dir_name
        for task in self.tasks:
            if os.path.exists(os.path.join(data_dir, task, question_dir)):
                q_list = os.listdir(os.path.join(data_dir, task, question_dir))
                i_list = os.listdir(os.path.join(data_dir, task, image_dir))
                q_prefix = os.path.join(data_dir, task, question_dir)
                i_prefix = os.path.join(data_dir, task, image_dir)
            else:
                fn_list = os.listdir(os.path.join(data_dir, task))
                q_list = [fn for fn in fn_list if '.txt' in fn]
                i_list = [fn for fn in fn_list if fn not in q_list]
                q_prefix = i_prefix = os.path.join(data_dir, task)

            q_list.sort()
            i_list.sort()
            assert len(q_list) == len(i_list)
            for q_fn, i_fn in zip(q_list, i_list):
                assert q_fn.split('.')[0] == i_fn.split('.')[0]
                q_path = os.path.join(q_prefix, q_fn)
                image_path = os.path.join(i_prefix, i_fn)
                with open(q_path, 'r') as f:
                    q1, a1 = f.readline().strip().split('\t')
                    q2, a2 = f.readline().strip().split('\t')
                self.data_list.append({
                    'img_path': image_path,
                    'question': q1,
                    'answer': a1,
                    'task': task
                })
                self.data_list.append({
                    'img_path': image_path,
                    'question': q2,
                    'answer': a2,
                    'task': task
                })

    def __len__(self) -> None:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        data_sample = self.data_list[idx]
        data_sample = self.pipeline(data_sample)
        return data_sample
