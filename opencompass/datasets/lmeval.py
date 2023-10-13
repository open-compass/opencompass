from datasets import Dataset, DatasetDict

from opencompass.datasets import BaseDataset


class LMEvalDataset(BaseDataset):
    """A dataset wrapper around the evaluator inputs, designed for
    OpenCompass's internal use."""

    @staticmethod
    def load(**kwargs):
        content = {k: v for k, v in kwargs.items() if v}
        return DatasetDict(dict(test=Dataset.from_dict(content)))
