from datasets import concatenate_datasets, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class XLSUMDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        path = kwargs.get('path', None)
        lans = [
            'oromo', 'french', 'amharic', 'arabic', 'azerbaijani', 'bengali',
            'burmese', 'chinese_simplified', 'chinese_traditional', 'welsh',
            'english', 'kirundi', 'gujarati', 'hausa', 'hindi', 'igbo',
            'indonesian', 'japanese', 'korean', 'kyrgyz', 'marathi', 'spanish',
            'scottish_gaelic', 'nepali', 'pashto', 'persian', 'pidgin',
            'portuguese', 'punjabi', 'russian', 'serbian_cyrillic',
            'serbian_latin', 'sinhala', 'somali', 'swahili', 'tamil', 'telugu',
            'thai', 'tigrinya', 'turkish', 'ukrainian', 'urdu', 'uzbek',
            'vietnamese', 'yoruba'
        ]

        datasets = []
        for lan in lans:
            dataset = load_dataset(path, lan)['validation']
            datasets.append(dataset)

        combined_dataset = concatenate_datasets(datasets)

        return combined_dataset
