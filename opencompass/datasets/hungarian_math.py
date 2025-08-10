import pandas as pd
from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class HungarianExamMathDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        df = pd.read_csv(path)
        df.columns = ['question']
        outputs = [{
            'question': question
        } for question in df['question'].tolist()]
        dataset = Dataset.from_list(outputs)
        return dataset
