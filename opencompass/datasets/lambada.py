import json
import re
import string
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path
from opencompass.utils.text_postprocessors import general_postprocess

from .base import BaseDataset


@LOAD_DATASET.register_module()
class lambadaDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(path)
            return dataset
        else:
            dataset = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    dataset.append(json.loads(line))
            dataset = Dataset.from_list(dataset)
            return DatasetDict({'test': dataset})


@ICL_EVALUATORS.register_module()
class LambadaEvaluator(BaseEvaluator):

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        score = 0.0
        for pred, refer in zip(predictions, references):
            pred = pred.strip().split(' ')[0]
            pred = re.split(f'[{string.punctuation}]', pred)[0]
            score += general_postprocess(pred) == general_postprocess(refer)
        score = 100.0 * score / len(predictions)
        return dict(accuracy=score)
