from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Union

import numpy as np
from scipy.stats import hypergeom

from opencompass.registry import ICL_EVALUATORS

from .icl_base_evaluator import BaseEvaluator


def compute_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _compute_pass_at_k(n, c, k, m):
    if m > min(c, k) or k > n or c < 0 or n <= 0 or m < 0:
        return 0.0
    return hypergeom.sf(m - 1, n, c, k)


def compute_g_pass_at_k(n, c, k, t):
    m = max(int(np.ceil(k * t)), 1)
    return _compute_pass_at_k(n, c, k, m)


def compute_mg_pass_at_k(n, c, k):
    l, r = int(np.ceil(k * 0.5)), k

    mg_pass_at_k = 0.0
    for i in range(l + 1, r + 1):
        mg_pass_at_k += _compute_pass_at_k(n, c, k, i)
    mg_pass_at_k = 2 * mg_pass_at_k / k

    return mg_pass_at_k


@ICL_EVALUATORS.register_module()
class GPassKEvaluator(BaseEvaluator):
    """Evaluator for computing G-Pass@k Metric.

    This Evaluator will firstly invoke task-specific `preprocess` on
    predictions to get a  consistency label for each prediction and
    the corresponding reference, and then metrics are calculated.

    This evaluator require the test dataset contains following keys:
        - subdivision: the name of subdivision or dataset split,
        - idx: the idx of each example.

    Args:

        k: core parameter for G-Pass@k, can be single integer or a
        list of integer, for example, k=16 will compute G-Pass@16,
        and [4, 8, 16] will compute G-Pass@{4,8,16}.

        replication: parameter to control the number of generations
        for estimating G-Pass@k, shoulde be a single integer. The
        total number of generations will be set to `k` * `replication`.

        thresholds: a list of float to controld the threshold in
        G-Pass@k.
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
    def preprocess(self, predictions, references, origin_prompt,
                   test_set) -> None:
        """Perform operations on predictions before computing metrics, for
        example, do answer_extraction and model_judge in mathematical reasoning
        task.

        Return:
            labels: A list contains the label which indicates whether
            prediction is consistency with reference at each position.
        """

    def score(self, predictions, references, origin_prompt,
              test_set) -> Dict[str, Any]:
        """Compute G-Pass@k metrics.

        Return:
            A dict contains overall metrics, metrics for each subdivision
            in test set, and details for each example in test set.
            like
            {'details': details for each example, 'G-Pass@16': xxx}
        """
        labels = self.preprocess(predictions, references, origin_prompt,
                                 test_set)

        example2replications = {}
        for example, label, prediction in zip(test_set, labels, predictions):
            example_abbr = f"{example['subdivision']}_{example['idx']}"
            if example_abbr not in example2replications:
                example2replications[example_abbr] = []
            example.update({'prediction': prediction, 'label': label})
            example2replications[example_abbr].append(example)
        for _, replications in example2replications.items():
            assert len(replications) == self.n, print(len(replications),
                                                      self.n)

        details = []
        all_dataset = set()
        total_pass_num, count = 0, 0
        for example_abbr, examples in example2replications.items():
            detail = {
                'question': examples[0]['question'],
                'answer': examples[0]['answer'],
                'question_type': examples[0]['question_type'],
                'options': examples[0]['options'],
                'subdivision': examples[0]['subdivision'],
                'idx': examples[0]['idx'],
                'prompt': examples[0]['prompt'],
                'predictions': [example['prediction'] for example in examples],
                'labels': [example['label'] for example in examples],
            }

            all_dataset.add(examples[0]['subdivision'])
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

        g_passk_details = OrderedDict()
        g_passk_details['details'] = details

        for k in self.k:
            for subdivision in sorted(list(all_dataset)):
                for threshold in self.thresholds:
                    g_passk_details[
                        f'{subdivision}/G-Pass@{k}_{threshold}'] = \
                            100. * np.mean(
                            [
                                detail[f'G-Pass@{k}_{threshold}']
                                for detail in details
                                if detail['subdivision'] == subdivision
                            ])
                g_passk_details[f'{subdivision}/mG-Pass@{k}'] = 100. * np.mean(
                    [
                        detail[f'mG-Pass@{k}'] for detail in details
                        if detail['subdivision'] == subdivision
                    ])

            for threshold in self.thresholds:
                g_passk_details[f'G-Pass@{k}_{threshold}'] = 100. * np.mean(
                    [detail[f'G-Pass@{k}_{threshold}'] for detail in details])
            g_passk_details[f'mG-Pass@{k}'] = 100. * np.mean(
                [detail[f'mG-Pass@{k}'] for detail in details])

        return g_passk_details
