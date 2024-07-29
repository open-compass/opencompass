import json

from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class TNewsDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        tnews_targets = {
            'news_agriculture': '农业新闻',
            'news_travel': '旅游新闻',
            'news_game': '游戏新闻',
            'news_tech': '科技类别公司新闻',
            'news_sports': '体育类别新闻',
            'news_edu': '初升高教育新闻',
            'news_entertainment': '娱乐圈新闻',
            'news_finance': '投资资讯',
            'news_military': '军事类别常识',
            'news_car': '车辆新闻',
            'news_house': '楼市新闻',
            'news_world': '环球不含中国类别新闻',
            'news_culture': '书籍文化历史类别新闻',
            'news_story': '故事类别新闻',
            'news_stock': '股票市场类别新闻',
        }
        if 'data_files' in kwargs:
            kwargs['data_files'] = get_data_path(kwargs['data_files'],
                                                 local_mode=True)
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            label_desc = example['label_desc']
            label_desc2 = tnews_targets[label_desc]
            example['label_desc2'] = label_desc2
            return example

        dataset = dataset.map(preprocess)
        return dataset


@LOAD_DATASET.register_module()
class TNewsDatasetV2(BaseDataset):

    @staticmethod
    def load(path):
        tnews_targets = {
            'news_agriculture': 'A',
            'news_travel': 'B',
            'news_game': 'C',
            'news_tech': 'D',
            'news_sports': 'E',
            'news_edu': 'F',
            'news_entertainment': 'G',
            'news_finance': 'H',
            'news_military': 'I',
            'news_car': 'J',
            'news_house': 'K',
            'news_world': 'L',
            'news_culture': 'M',
            'news_story': 'N',
            'news_stock': 'O',
        }
        path = get_data_path(path, local_mode=True)
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                item = {
                    'sentence': line['sentence'],
                    'label_desc2': tnews_targets[line['label_desc']],
                }
                data.append(item)
        return Dataset.from_list(data)
