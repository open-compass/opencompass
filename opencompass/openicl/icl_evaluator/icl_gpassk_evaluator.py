from abc import abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from scipy.stats import hypergeom

from opencompass.registry import ICL_EVALUATORS

from .icl_base_evaluator import BaseEvaluator


def compute_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _compute_g_pass_at_k(n, c, k, m):
    if m > min(c, k) or k > n or c < 0 or n <= 0 or m < 0:
        return 0.0
    return hypergeom.sf(m - 1, n, c, k)


def compute_g_pass_at_k(n, c, k, t):
    m = max(int(np.ceil(k * t)), 1)
    return _compute_g_pass_at_k(n, c, k, m)


def compute_mg_pass_at_k(n, c, k):
    l, r = int(np.ceil(k * 0.5)), k

    mg_pass_at_k = 0.0
    for i in range(l + 1, r + 1):
        mg_pass_at_k += _compute_g_pass_at_k(n, c, k, i)
    mg_pass_at_k = 2 * mg_pass_at_k / k

    return mg_pass_at_k


@ICL_EVALUATORS.register_module()
class GPassKEvaluator(BaseEvaluator):
    """Evaluator for computing the G-Pass@k Metric.

    This evaluator performs the following steps:
    1. Invokes task-specific `preprocess` on predictions to
    assign a consistency label to each prediction and its
    corresponding reference.
    2. Calculates metrics for each input example based on
    these labels.
    3. Aggregates the overall metrics through a task-specific
    `postprocess`.

    Args:
        k (int or list of int): Number of predictions to be
        considered in G-Pass@k. It can be a single integer
        (e.g., `k=16` computes G-Pass@16) or a list of
        integers (e.g., `[4, 8, 16]` computes G-Pass@4,
        G-Pass@8, and G-Pass@16).

        replication (int): Controls the number of generations
        used to estimate G-Pass@k. The total number of
        generations is determined by multiplying the
        maximum of `k` with `replication`. This parameter
        should be a single integer.

        thresholds (list of float): A list of floating-point
        numbers that define the thresholds for the G-Pass@k
        metric.
    """

    def __init__(
            self,
            k: Union[int, List[int]] = 16,
            replication: int = 3,
            thresholds: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]) -> None:
        super().__init__()

        if isinstance(k, int):
            k = [k]

        self.k = k
        self.replication = replication
        self.n = max(k) * replication
        self.thresholds = thresholds

    @property
    def output_dir(self):
        # please see opencompass/opencompass/tasks/openicl_eval.py Line 197-200
        return self._out_dir

    @abstractmethod
    def preprocess(self, predictions, references, test_set) -> None:
        """Perform operations on predictions before computing metrics, for
        example, do answer_extraction and model_judge in mathematical reasoning
        task.

        Return:
            labels: A list contains the label which indicates whether
            prediction is consistency with reference at each position.
        """
        raise NotImplementedError

    @abstractmethod
    def group(self, predictions, labels, test_set) -> Dict[str, Any]:
        """Group the predictions and references.

        Return:
            A dict contains the grouped predictions and references.
        """
        raise NotImplementedError

    @abstractmethod
    def reduce(self, details) -> Dict[str, Any]:
        """Aggregate the overall metrics.

        Return:
            A dict contains overall metrics, like:
            {'details': details for each example, 'G-Pass@16': xxx}
        """
        raise NotImplementedError

    def score(self, predictions, references, test_set) -> Dict[str, Any]:
        """Compute G-Pass@k metrics.

        Return:
            A dict contains  metrics for each dataset sample and
            overall metrics reduced by `self.reduce`, like:
            {'details': details for each example, 'G-Pass@16': xxx}
        """
        labels = self.preprocess(predictions, references, test_set)
        grouped_examples = self.group(predictions, labels, test_set)

        details = []
        total_pass_num, count = 0, 0
        for example_abbr, examples in grouped_examples.items():
            detail = {
                k: v
                for k, v in examples[0].items()
                if k not in ['prediction', 'label']
            }
            detail.update({
                'predictions': [{
                    'prediction': example['prediction'],
                    'label': example['label']
                } for example in examples],
            })

            current_example_labels = [e['label'] for e in examples]
            c = int(np.sum(current_example_labels))

            for k in self.k:
                for threshold in self.thresholds:
                    detail[f'G-Pass@{k}_{threshold}'] = compute_g_pass_at_k(
                        n=self.n, c=c, k=k, t=threshold)
                detail[f'mG-Pass@{k}'] = compute_mg_pass_at_k(n=self.n,
                                                              c=c,
                                                              k=k)
            count += self.n
            total_pass_num += c

            details.append(detail)

        return self.reduce(details)
