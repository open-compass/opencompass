"""Binary Bayes@N posterior moments for repeated evaluation.

See Algorithm 1 in https://arxiv.org/abs/2510.04265.
"""

import math
from numbers import Real

import numpy as np


def validate_bayes_at_n_dataset(n, dataset):
    """Require the same nonempty question set exactly once per replica."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('Bayes@N requires a positive integer n')
    if not len(dataset) or len(dataset) % n:
        raise ValueError('Bayes@N requires a nonempty dataset divisible by n')
    size = len(dataset) // n
    first = None
    for start in range(0, len(dataset), size):
        identities = []
        for example in dataset.select(range(start, start + size)):
            if ('subdivision' not in example or 'idx' not in example
                    or not isinstance(example['subdivision'], str)):
                raise ValueError(
                    'Bayes@N requires subdivision and idx metadata')
            identities.append((example['subdivision'], example['idx']))
        try:
            identities = set(identities)
        except TypeError as exc:
            raise ValueError('Bayes@N requires hashable question IDs') from exc
        if len(identities) != size or (first is not None
                                       and identities != first):
            raise ValueError('Bayes@N requires each question once per replica')
        first = identities


def compute_bayes_at_n(n, details, dataset):
    """Return corpus and subdivision mean/SD in percentage points."""
    if len(details) != len(dataset):
        raise ValueError(
            'Bayes@N requires one correctness detail per prediction')
    successes = {}
    for detail, example in zip(details, dataset):
        if not isinstance(detail, dict) or detail.get('error'):
            raise ValueError('Bayes@N requires error-free correctness details')
        value = next((detail[key]
                      for key in ('correct', 'is_correct', 'cascade_correct')
                      if detail.get(key) is not None), None)
        if (not isinstance(value, (Real, np.bool_)) or not math.isfinite(value)
                or value not in (0, 1)):
            raise ValueError('Bayes@N requires binary correctness for every '
                             'prediction')
        key = (example['subdivision'], example['idx'])
        successes[key] = successes.get(key, 0) + int(value)

    def moments(counts):
        size = len(counts)
        total = n + 2
        mean = math.fsum((count + 1) / total for count in counts) / size
        variance = math.fsum(
            (count + 1) * (n - count + 1) / (total**2 * (total + 1))
            for count in counts)
        return {
            f'Bayes@{n}': 100.0 * mean,
            f'Bayes@{n} posterior SD': 100.0 * math.sqrt(variance) / size,
        }

    result = moments(list(successes.values()))
    for subdivision in sorted({key[0] for key in successes}):
        counts = [v for (sub, _), v in successes.items() if sub == subdivision]
        result.update(
            {f'{subdivision}/{k}': v
             for k, v in moments(counts).items()})
    return result
