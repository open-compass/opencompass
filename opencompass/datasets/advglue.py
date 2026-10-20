import json
from typing import List, Union

from datasets import Dataset, concatenate_datasets

from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils import get_data_path

from .base import BaseDataset


class AdvDataset(BaseDataset):
    """Base adv GLUE dataset. Adv GLUE is built on GLUE dataset. The main
    purpose is to eval the accuracy drop on original set and adv set.

    Args:
        subset (str): The subset task of adv GLUE dataset.
        filter_keys (str): The keys to be filtered to create the original
            set for comparison.
    """

    def __init__(
        self,
        subset: str,
        filter_keys: Union[str, List[str]],
        **kwargs,
    ):
        self.subset = subset
        if isinstance(filter_keys, str):
            filter_keys = [filter_keys]
        self.filter_keys = filter_keys
        super().__init__(**kwargs)

    def aug_with_original_data(self, dataset):
        """Create original dataset and concat to the end."""
        # Remove data without original reference
        dataset = dataset.filter(
            lambda x: any([x[k] for k in self.filter_keys]))

        def ori_preprocess(example):
            for k in self.filter_keys:
                if example[k]:
                    new_k = k.split('original_')[-1]
                    example[new_k] = example[k]
                    example['type'] = 'original'
            return example

        original_dataset = dataset.map(ori_preprocess)

        return concatenate_datasets([dataset, original_dataset])

    def load(self, path):
        """Load dataset and aug with original dataset."""

        path = get_data_path(path)
        with open(path, 'r') as f:
            raw_data = json.load(f)
            subset = raw_data[self.subset]

            # In case the missing keys in first example causes Dataset
            # to ignore them in the following examples when building.
            for k in self.filter_keys:
                if k not in subset[0]:
                    subset[0][k] = None

            dataset = Dataset.from_list(raw_data[self.subset])

        dataset = self.aug_with_original_data(dataset)

        def choices_process(example):
            example['label_option'] = chr(ord('A') + example['label'])
            return example

        dataset = dataset.map(choices_process)
        return dataset


# label 0 for A. negative
# label 1 for B. positive
class AdvSst2Dataset(AdvDataset):
    """Adv GLUE sst2 dataset."""

    def __init__(self, **kwargs):
        super().__init__(subset='sst2',
                         filter_keys='original_sentence',
                         **kwargs)


# label 0 for not_duplicate, A. no
# label 1 for duplicate, B. yes
class AdvQqpDataset(AdvDataset):
    """Adv GLUE qqp dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            subset='qqp',
            filter_keys=['original_question1', 'original_question2'],
            **kwargs)


# # label 0 for entailment, A. yes
# # label 1 for neutral, B. maybe
# # label 2 for contradiction, C. no
class AdvMnliDataset(AdvDataset):
    """Adv GLUE mnli dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            subset='mnli',
            filter_keys=['original_premise', 'original_hypothesis'],
            **kwargs)


# # label 0 for entailment, A. yes
# # label 1 for neutral, B. maybe
# # label 2 for contradiction, C. no
class AdvMnliMMDataset(AdvDataset):
    """Adv GLUE mnli mm dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            subset='mnli-mm',
            filter_keys=['original_premise', 'original_hypothesis'],
            **kwargs)


# # label 0 for entailment, A. yes
# # label 1 for not entailment, B. no
class AdvQnliDataset(AdvDataset):
    """Adv GLUE qnli dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            subset='qnli',
            filter_keys=['original_question', 'original_sentence'],
            **kwargs)


# # label 0 for entailment, A. yes
# # label 1 for not entailment, B. no
class AdvRteDataset(AdvDataset):
    """Adv GLUE rte dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            subset='rte',
            filter_keys=['original_sentence1', 'original_sentence2'],
            **kwargs)


class AccDropEvaluator(AccEvaluator):
    """Eval accuracy drop."""

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

        n = len(predictions)
        assert n % 2 == 0, 'Number of examples should be even.'
        acc_after = super().score(predictions[:n // 2], references[:n // 2])
        acc_before = super().score(predictions[n // 2:], references[n // 2:])
        acc_drop = 1 - acc_after['accuracy'] / acc_before['accuracy']
        return dict(acc_drop=acc_drop,
                    acc_after=acc_after['accuracy'],
                    acc_before=acc_before['accuracy'])
