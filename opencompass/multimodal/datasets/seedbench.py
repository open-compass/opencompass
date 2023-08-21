import importlib
import json
import os.path as osp
from typing import List

import numpy as np
import torch
from decord import VideoReader, cpu
from mmengine.dataset import Compose
from PIL import Image
from torch.utils.data import Dataset

from opencompass.registry import DATASETS


@DATASETS.register_module()
class SEEDBenchDataset(Dataset):
    """Dataset to load SEED-Bench dataset.

    Args:
        ann_file (str): The path of the annotation file.
        cc3m_path (str): The data path of the image dimension(1-9).
        sthv2_path (str): The data path of the dimension 10.
        epic_kitchens_path (str): The data path of the dimension 11.
        breakfast_path (str): The data path of the dimension 12.
        image_pipeline (List[dict]): The data transforms for image.
        video_pipeline (List[dict]): The data transforms for video.
        only_image (bool): Whether run SEED-Bench only with image data.
            Defaults to True.
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
        only_image: bool = True,
    ) -> None:
        ann_file = json.load(open(ann_file, 'rb'))
        if 'questions' in ann_file.keys():
            self.ann_file = ann_file['questions']
        self.cc3m_path = cc3m_path
        self.sthv2_path = sthv2_path
        self.epic_kitchens_path = epic_kitchens_path
        self.breakfast_path = breakfast_path
        self.image_pipeline = Compose(image_pipeline)
        if only_image:
            image_ann_file = [
                ann for ann in self.ann_file if ann['data_type'] == 'image'
            ]
            self.ann_file = image_ann_file
        if not only_image:
            raise NotImplementedError
            self.video_pipeline = Compose(video_pipeline)

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
            raw_image = Image.open(open(data_path, 'rb')).convert('RGB')
            data['data_path'] = data_path
            data['img'] = raw_image
            data = self.image_pipeline(data)
        elif item['data_type'] == 'video':
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
                raise ValueError('The question type id is not valid.')

            # preprocessing videos in evaluation dimension 10-12
            use_pyav = False
            if 'segment' in data.keys():
                segment = data['segment']
                if isinstance(segment[0], int):
                    # using pyav for decoding videos in evaluation dimension 12
                    use_pyav = True
                start, end = segment[0], segment[1]
            else:
                start = 0.0
                end = 0.0

            if use_pyav:
                # using pyav for videos in evaluation dimension 12
                av = importlib.importmodule('av')
                reader = av.open(data_path)
                frames = [
                    torch.from_numpy(f.to_rgb().to_ndarray())
                    for f in reader.decode(video=0)
                ]
                video_len = len(frames)
                start_frame, end_frame = start, end
                end_frame = min(end_frame, video_len)
                offset = self.get_index(end_frame - start_frame, 8)
                frame_indices = offset + start_frame
                buffer = torch.stack([frames[idx] for idx in frame_indices])
                buffer = buffer.numpy()
            else:
                # using decord for videos in evaluating dimension 10-11
                import io

                import mmengine.fileio as fileio
                file_obj = io.BytesIO(fileio.get(data_path))
                vr = VideoReader(file_obj, num_threads=1, ctx=cpu(0))
                video_len = len(vr)
                fps = vr.get_avg_fps()
                if 'segment' in data.keys():
                    # obtain start and end frame for the video segment
                    # in evaluation dimension 11
                    start_frame = int(min(max(start * fps, 0), video_len - 1))
                    end_frame = int(min(max(end * fps, 0), video_len - 1))
                    tot_frames = int(end_frame - start_frame)
                    offset = self.get_index(tot_frames, 8)
                    frame_indices = offset + start_frame
                else:
                    # sample frames of the video in evaluation dimension 10
                    frame_indices = self.get_index(video_len - 1, 8)
                vr.seek(0)
                buffer = vr.get_batch(frame_indices)
                buffer = buffer.asnumpy()
            data['imgs'] = buffer
            data = self.video_pipeline(data)

        else:
            raise ValueError('The data type is not valid.')

        return data

    def get_index(self, num_frames, num_segments):
        if num_segments > num_frames:
            offsets = np.array([idx for idx in range(num_frames)])
        else:
            # uniform sampling
            seg_size = float(num_frames - 1) / num_segments
            start = int(seg_size / 2)
            offsets = np.array([
                start + int(np.round(seg_size * idx))
                for idx in range(num_segments)
            ])
        return offsets
