import re
from typing import List

from datasets import load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CrowspairsDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        dataset = load_dataset(**kwargs)

        def preprocess(example):
            example['label'] = 0
            return example

        return dataset.map(preprocess)


@LOAD_DATASET.register_module()
class CrowspairsDatasetV2(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            example['label'] = 'A'
            return example

        return dataset.map(preprocess)


def crowspairs_postprocess(text: str) -> str:
    """Cannot cover all the cases, try to be as accurate as possible."""
    if re.search('Neither', text) or re.search('Both', text):
        return 'invalid'

    if text != '':
        first_option = text[0]
        if first_option.isupper() and first_option in 'AB':
            return first_option

        if re.search(' A ', text) or re.search('A.', text):
            return 'A'

        if re.search(' B ', text) or re.search('B.', text):
            return 'B'

    return 'invalid'


class CrowspairsEvaluator(BaseEvaluator):
    """Calculate accuracy and valid accuracy according the prediction for
    crows-pairs dataset."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions: List, references: List) -> dict:
        """Calculate scores and accuracy.

        Args:
            predictions (List): List of probabilities for each class of each
                sample.
            references (List): List of target labels for each sample.

        Returns:
            dict: calculated scores.
        """
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length.'
            }
        all_match = 0
        for i, j in zip(predictions, references):
            all_match += i == j

        valid_match = 0
        valid_length = 0
        for i, j in zip(predictions, references):
            if i != 'invalid':
                valid_length += 1
                valid_match += i == j

        accuracy = round(all_match / len(predictions), 4) * 100
        valid_accuracy = round(valid_match / valid_length, 4) * 100
        valid_frac = round(valid_length / len(predictions), 4) * 100
        return dict(accuracy=accuracy,
                    valid_accuracy=valid_accuracy,
                    valid_frac=valid_frac)
