from typing import List, Optional

from datasets import Dataset, DatasetDict

from opencompass.datasets import BaseDataset


class LMEvalDataset(BaseDataset):
    """A dataset wrapper around the evaluator inputs, designed for
    OpenCompass's internal use."""

    @staticmethod
    def load(predictions: List, references: Optional[List] = None):
        content = {'prediction': predictions}
        if references:
            content['reference'] = references
        return DatasetDict(dict(test=Dataset.from_dict(content)))
