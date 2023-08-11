import os.path as osp
import re
import tempfile
from typing import List
from datasets import DatasetDict, load_dataset

from .base import BaseDataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator

class HumanevalXDataset(BaseDataset):

    @staticmethod
    def load(language='python', **kwargs):
        dataset = load_dataset(**kwargs, split='test')
        return dataset
    

if __name__ == "__main__":
    pass