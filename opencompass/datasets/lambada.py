import re
import string

from datasets import DatasetDict, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils.text_postprocessors import general_postprocess

from .base import BaseDataset


@LOAD_DATASET.register_module()
class lambadaDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs, split='test')

        def preprocess(example):
            prompt, target = example['text'].strip().rsplit(' ', 1)
            example['prompt'] = prompt
            example['label'] = target
            return example

        dataset = dataset.map(preprocess)
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
