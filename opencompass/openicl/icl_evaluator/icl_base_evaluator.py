"""Base Evaluator."""

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Union

import numpy as np
from datasets import Dataset
from scipy.stats import hypergeom


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


class BaseEvaluator:

    def __init__(self) -> None:
        pass

    @property
    def output_dir(self):
        # please see opencompass/opencompass/tasks/openicl_eval.py Line 197-200
        return self._out_dir

    def group(self, n: int, details: List[Dict[str, Any]],
              test_set: Dataset) -> Dict[str, Any]:
        example2replications = {}
        for detail, example in zip(details, test_set):
            example_abbr = f"{example['subdivision']}_{example['idx']}"
            if example_abbr not in example2replications:
                example2replications[example_abbr] = []
            example.update({'detail': detail})
            example2replications[example_abbr].append(example)
        for _, replications in example2replications.items():
            assert len(replications) == n, print(len(replications), n)
        return example2replications

    def reduce(self, details: List[Dict[str, Any]]) -> Dict[str, Any]:
        g_passk_details = OrderedDict()
        all_subdivisions = set(
            [detail['example_abbr'].split('_')[0] for detail in details])
        all_metrics = list(details[0].keys())

        for subdivision in sorted(list(all_subdivisions)):
            for metric in all_metrics:
                if metric in ['predictions', 'example_abbr']:
                    continue
                g_passk_details[f'{subdivision}/{metric}'] = 100 * np.mean([
                    detail[metric] for detail in details
                    if detail['example_abbr'].split('_')[0] == subdivision
                ])

        for metric in all_metrics:
            if metric in ['predictions', 'example_abbr']:
                continue
            g_passk_details[metric] = 100.0 * np.mean(
                [detail[metric] for detail in details])
        return g_passk_details

    def evaluate(
        self,
        k: Union[int, List[int]],
        n: int,
        original_dataset: Dataset,
        **score_kwargs,
    ):
        # Check if predictions and references have the
        # same length if both are provided
        if ('predictions' in score_kwargs and 'references' in score_kwargs
                and score_kwargs['references'] is not None):
            if len(score_kwargs['predictions']) != len(
                    score_kwargs['references']):
                raise ValueError(
                    'Predictions and references must have the same length')

        real_size = len(original_dataset) // n
        all_details = []
        all_results = []
        for i in range(n):

            def select_fn(i, real_size, x):
                if isinstance(x, Dataset):
                    return x.select(range(i * real_size, (i + 1) * real_size))
                elif isinstance(x, Iterable):
                    return x[i * real_size:(i + 1) * real_size]
                else:
                    return x

            results = self.score(
                **{
                    key: select_fn(i, real_size, value)
                    for key, value in score_kwargs.items()
                })
            details = results.pop('details', None)
            if details is not None:
                if isinstance(details, Dict):
                    details = list(details.values())
                all_details.extend(details)
            all_results.append(results)

        eval_results = {}
        for single_results in all_results:
            for key in single_results:
                if key not in eval_results:
                    eval_results[key] = []
                eval_results[key].append(single_results[key])
        for key in deepcopy(eval_results):
            if isinstance(eval_results[key][0], float) or isinstance(
                    eval_results[key][0], int):
                if n > 1:
                    eval_results[key + f' ({n} runs average)'] = np.mean(
                        eval_results[key])
                    eval_results.pop(key)
                else:
                    eval_results[key] = np.mean(eval_results[key])
            else:
                eval_results[key] = eval_results[key][0]

        grouped_examples = self.group(n, all_details, original_dataset)
        can_calculate = False
        if len(all_details) != 0:
            eval_details = []
            for example_abbr, examples in grouped_examples.items():
                detail = {'predictions': [], 'example_abbr': example_abbr}

                c = 0
                for example in examples:
                    detail['predictions'].append(example['detail'])
                    # only compute G-Pass@k when details have correct labels
                    if example['detail'].get('correct', None) is not None:
                        can_calculate = True
                        c += int(example['detail']['correct'])
                    elif example['detail'].get('is_correct', None) is not None:
                        can_calculate = True
                        c += int(example['detail']['is_correct'])

                k_list = [k] if isinstance(k, int) else k
                if can_calculate and n > 1 and max(k_list) > 1:
                    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
                    for _k in k_list:
                        for threshold in thresholds:
                            g_pass = compute_g_pass_at_k(n=n,
                                                         c=c,
                                                         k=_k,
                                                         t=threshold)
                            detail[f'G-Pass@{_k}_{threshold}'] = g_pass
                        detail[f'mG-Pass@{_k}'] = compute_mg_pass_at_k(n=n,
                                                                       c=c,
                                                                       k=_k)

                eval_details.append(detail)

            if can_calculate and n > 1 and max(k_list) > 1:
                eval_results.update(self.reduce(eval_details))

            # Store eval_details in eval_results
            eval_results['details'] = eval_details

            # Process details to flatten the predictions
            for detail in eval_details:
                # Extract all prediction fields and flatten them
                flattened_predictions = {}
                for pred in detail['predictions']:
                    for k, v in pred.items():
                        if k not in flattened_predictions:
                            flattened_predictions[k] = [v]
                        else:
                            flattened_predictions[k].append(v)

                # Replace the predictions list with the flattened dictionary
                for k, v in flattened_predictions.items():
                    detail[k] = v

                # Remove the original predictions field
                detail.pop('predictions')
            return eval_results

        # If there are no details, return results
        return results

    def score(self):
        raise NotImplementedError("Method hasn't been implemented yet")

    @staticmethod
    def is_num_equal(predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        else:
            return
