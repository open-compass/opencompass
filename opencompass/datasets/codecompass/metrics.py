import numpy as np


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        import itertools
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        num_samples_it = iter(num_samples)

    return np.array([
        estimator(int(n), int(c), k)
        for n, c in zip(num_samples_it, num_correct)
    ])


def compute_metrics_from_results(results: dict, k_list=[1]):
    print('\n--- [DEBUG] compute_metrics_from_results started ---')
    print(f'  > Received results for {len(results)} problems.')

    total = []
    correct = []

    if not results:
        metrics = {}
        for k in k_list:
            metrics[f'pass@{k}'] = 0.0
        if 1 in k_list or not k_list:
            metrics['details'] = {'pass@1': []}
        return metrics

    for problem_idx in sorted(results.keys()):
        problem_results = results[problem_idx]
        if not problem_results:
            total.append(0)
            correct.append(0)
            continue

        total.append(len(problem_results))

        num_correct_generations = 0
        for gen_result in problem_results:
            if not gen_result:
                continue
            if all(res > 0 for res in gen_result):
                num_correct_generations += 1
        correct.append(num_correct_generations)

    total_arr = np.array(total)
    correct_arr = np.array(correct)

    metrics = {}
    # For details, a problem is "correct" if at least one generation passed.
    pass_1_details = (correct_arr > 0).astype(float).tolist()

    for k in k_list:
        if np.all(total_arr >= k):
            pass_k_mean = estimate_pass_at_k(total_arr, correct_arr, k).mean()
            metrics[f'pass@{k}'] = pass_k_mean
        else:
            metrics[f'pass@{k}'] = np.nan

    # Add the detailed pass@1 results for building the final report
    if 1 in k_list or not k_list:
        metrics['details'] = {'pass@1': pass_1_details}

    return metrics
