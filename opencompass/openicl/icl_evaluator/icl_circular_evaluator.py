import collections

from opencompass.registry import ICL_EVALUATORS

from .icl_base_evaluator import BaseEvaluator


@ICL_EVALUATORS.register_module()
class CircularEvaluator(BaseEvaluator):
    """Robust circular evaluator for multi-choice questions."""

    def __init__(self) -> None:
        super().__init__()
        self.cp4 = ['ABCD', 'BCDA', 'CDAB', 'DABC']
        self.cp1 = ['ABCD']

    def score(self, predictions, references):
        """Calculate the accuracy of predictions.

        Args:
            predictions (list): List of predictions.
            references (list): List of references.

        Returns:
            dict: A dict of evaluation results.
        """
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        self._metrics = {}
        self._metrics.update({'acc_4': 0, 'acc_1': 0})
        # Accuracy for patterns with no circular shift / 4 circular shifts
        for pred, reference in zip(predictions, references):
            index, ref, circular_pattern = reference.split('--')
            if circular_pattern in self.cp4:
                self._metrics['acc_4'] += 1 if pred == ref else 0
            if circular_pattern in self.cp1:
                self._metrics['acc_1'] += 1 if pred == ref else 0
        for k in ['acc_4', 'acc_1']:
            self._metrics[k] = self._metrics[k] / len(predictions) * 4 / int(
                k.split('_')[-1]) * 100

        # Accuracy for patterns with no circular shift / 4 circular shifts
        details = {4: {}, 1: {}}
        for pred, reference in zip(predictions, references):
            index, ref, circular_pattern = reference.split('--')
            if index not in details[4]:
                details[4][index] = []
                details[1][index] = []
            if circular_pattern in self.cp4:
                details[4][index].append(True if pred == ref else False)
            if circular_pattern in self.cp1:
                details[1][index].append(True if pred == ref else False)
        # Calculate accuracy for having at least j correct out of i total
        for i in [1, 4]:
            for j in range(0, i + 1):
                count, total = 0, 0
                for index in details[i]:
                    if sum(details[i][index]) >= j:
                        count += 1
                    total += 1
                self._metrics[f'more_{i}_{j}'] = count / total * 100
        # Consider fully correct as correct
        for i in [1, 4]:
            self._metrics[f'perf_{i}'] = self._metrics[f'more_{i}_{i}']

        # Calculate voting accuracy
        voting = {'vote_4': {}, 'vote_1': {}}
        refs = {}
        for pred, reference in zip(predictions, references):
            index, ref, circular_pattern = reference.split('--')
            c = circular_pattern
            back_map = {'A': c[0], 'B': c[1], 'C': c[2], 'D': c[3]}
            ref = back_map[ref]
            if pred not in ['A', 'B', 'C', 'D']:
                pred = '-'
            else:
                pred = back_map[pred]
            if index not in voting['vote_4']:
                voting['vote_4'][index] = collections.Counter()
                voting['vote_1'][index] = collections.Counter()
                refs[index] = ref

            if c in self.cp4:
                voting['vote_4'][index][pred] += 1
            if c in self.cp1:
                voting['vote_1'][index][pred] += 1
        for k in ['vote_4', 'vote_1']:
            voting_count = 0
            for index in voting[k]:
                if refs[index] == voting[k][index].most_common(1)[0][0]:
                    voting_count += 1
            self._metrics[k] = voting_count / len(voting[k]) * 100

        # Calculate the frequency of ABCD in model predictions
        prior_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, '-': 0}
        for pred, reference in zip(predictions, references):
            if pred in ['A', 'B', 'C', 'D']:
                prior_counts[pred] += 1
            else:
                prior_counts['-'] += 1
        for k in ['A', 'B', 'C', 'D', '-']:
            self._metrics[f'prior_{k}'] = prior_counts[k] / len(
                predictions) * 100

        return self._metrics
