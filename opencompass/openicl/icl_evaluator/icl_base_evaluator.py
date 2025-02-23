"""Base Evaluator."""
from typing import Union, List, Dict, Any, Iterable
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from scipy.stats import hypergeom
from datasets import Dataset


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

    def group(self, n: int, details: List[Dict[str, Any]], test_set: Dataset) -> Dict[str, Any]:
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
        all_subdivisions = set([detail['example_abbr'].split('_')[0] for detail in details])
        all_metrics = list(details[0].keys())
        
        for subdivision in sorted(list(all_subdivisions)):
            for metric in all_metrics:
                if metric in ['predictions', 'example_abbr']:
                    continue
                g_passk_details[f'{subdivision}/{metric}'] = 100 * np.mean([
                    detail[metric]
                    for detail in details
                    if detail['example_abbr'].split('_')[0] == subdivision
                ])

        for metric in all_metrics:
            if metric in ['predictions', 'example_abbr']:
                continue
            g_passk_details[metric] = 100. * np.mean([detail[metric] for detail in details])
        return g_passk_details

    def evaluate(self, k: Union[int, List[int]], 
                 repeat: int, test_set: Dataset,  **score_kwargs):
        n = (max(k) if isinstance(k, List) else k) * repeat
        print(len(score_kwargs['predictions']))
        real_size = len(test_set) // n
        all_details = []
        all_results = []
        for i in range(n):
            results = self.score(**{
                key: value[i * real_size: (i + 1) * real_size] if isinstance(value, Iterable) else value
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
            if isinstance(eval_results[key][0], float) or isinstance(eval_results[key][0], int):
                if n > 1:
                    eval_results[key + f' ({n // repeat}x{repeat}={n} runs average)'] = np.mean(eval_results[key])
                    eval_results.pop(key)
                else:
                    eval_results[key] = np.mean(eval_results[key])
            else:
                eval_results[key] = eval_results[key][0]

        grouped_examples = self.group(n, all_details, test_set)
        if len(all_details) != 0:
            eval_details = []
            for example_abbr, examples in grouped_examples.items():
                detail = {
                    'predictions': [],
                    'example_abbr': example_abbr
                }

                c = 0
                can_calculate = False
                for example in examples:
                    detail['predictions'].append(example['detail'])
                    # only compute G-Pass@k when details have correct labels
                    if example['detail'].get('correct', None) is not None:
                        can_calculate = True
                        c += int(example['detail']['correct'])
                    elif example['detail'].get('is_correct', None) is not None:
                        can_calculate = True
                        c += int(example['detail']['is_correct'])

                if can_calculate:
                    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
                    for _k in ([k] if isinstance(k, int) else k):
                        for threshold in thresholds:
                            detail[f'G-Pass@{_k}_{threshold}'] = compute_g_pass_at_k(
                                n=n, c=c, k=_k, t=threshold)
                        detail[f'mG-Pass@{_k}'] = compute_mg_pass_at_k(n=n, c=c, k=_k)

                eval_details.append(detail)

            eval_results.update(self.reduce(eval_details))
            eval_results['details'] = eval_details
        
        return eval_results

    def score(self):
        raise NotImplementedError("Method hasn't been implemented yet")

    @staticmethod
    def is_num_equal(predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        else:
            return
