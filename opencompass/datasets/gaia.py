import json
import os
from os import environ

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils.datasets_info import DATASETS_MAPPING

from .base import BaseDataset


@LOAD_DATASET.register_module()
class GAIADataset(BaseDataset):

    @staticmethod
    def load(path, local_mode: bool = False):
        rows = []
        if environ.get('DATASET_SOURCE') == 'HF':
            from datasets import load_dataset
            try:
                hf_id = DATASETS_MAPPING[path]['hf_id']
                # 因为ModelScope的GAIA数据集读取存在问题，所以从huggingface读取
                ds = load_dataset(hf_id, '2023_all', split='validation')
                rows = []
                for item in ds:
                    rows.append({
                        'question': item['Question'],
                        'answerKey': item['Final answer'],
                        'file_path': item['file_path'],
                        'file_name': item['file_name'],
                        'level': item['Level']
                    })
            except Exception as e:
                print(f'Error loading local file: {e}')
        else:
            # 从本地读取
            compass_data_cache = os.environ.get('COMPASS_DATA_CACHE')
            local_path = DATASETS_MAPPING[path]['local']
            local_path = os.path.join(compass_data_cache, local_path)
            with open(local_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line.strip())
                    # 构建数据行
                    row_data = {
                        'question': line['Question'],
                        'answerKey': line['Final answer'],
                        'file_name': line['file_name'],
                        'level': line['Level']
                    }

                    # 只有在file_name不为空时设置file_path
                    if line['file_name']:
                        file_name = line['file_name']
                        row_data['file_path'] = f'{local_path}/{file_name}'
                    else:
                        row_data['file_path'] = ''

                    rows.append(row_data)
        return Dataset.from_list(rows)
