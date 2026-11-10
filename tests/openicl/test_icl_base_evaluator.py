"""Tests for BaseEvaluator repeated-run (n > 1) aggregation semantics.

Guards against the silent degradation documented in
docs/en/extension/new_evaluator_and_summarizer.md: when not every score()
call returns complete, non-empty details, the base class must not quietly
fall back to the last batch's result.
"""

import unittest

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator


class ScriptedEvaluator(BaseEvaluator):
    """Returns a canned result for each score() call, in order."""

    def __init__(self, scripted_results):
        super().__init__()
        self.scripted_results = list(scripted_results)
        self.calls = 0

    def score(self, predictions, references=None, test_set=None):
        result = self.scripted_results[min(self.calls,
                                           len(self.scripted_results) - 1)]
        self.calls += 1
        return dict(result)


def make_dataset(n_replicas, replica_size):
    rows = []
    for _ in range(n_replicas):
        for i in range(replica_size):
            rows.append({'subdivision': 'test', 'idx': i})
    return Dataset.from_list(rows)


class TestRepeatedEvaluationDetails(unittest.TestCase):

    def _run(self, evaluator, n, size):
        return evaluator.evaluate(
            k=1,
            n=n,
            original_dataset=make_dataset(n, size),
            predictions=list(range(n * size)),
        )

    def test_metrics_averaged_when_details_missing(self):
        """n>1 without details: metrics must be the n-run mean, not the
        last batch's raw result (the documented silent degradation)."""
        evaluator = ScriptedEvaluator([
            {'accuracy': 60.0},
            {'accuracy': 80.0},
        ])
        results = self._run(evaluator, n=2, size=3)
        self.assertIn('accuracy (2 runs average)', results)
        self.assertAlmostEqual(results['accuracy (2 runs average)'], 70.0)

    def test_partial_details_skips_aggregation_without_crash(self):
        """n>1 with details from only some runs: must not raise; metrics
        are still averaged; incomplete detail aggregation is skipped."""
        evaluator = ScriptedEvaluator([
            {'accuracy': 60.0, 'details': [
                {'pred': i, 'answer': i, 'correct': True}
                for i in range(3)
            ]},
            {'accuracy': 80.0},  # no details in this run
        ])
        results = self._run(evaluator, n=2, size=3)
        self.assertAlmostEqual(results['accuracy (2 runs average)'], 70.0)
        self.assertNotIn('details', results)

    def test_full_details_aggregation_unchanged(self):
        """n>1 with complete details from every run: behavior preserved —
        metrics averaged, cross-run details grouped per sample."""
        evaluator = ScriptedEvaluator([
            {'accuracy': 60.0, 'details': [
                {'pred': i, 'answer': i, 'correct': i % 2 == 0}
                for i in range(3)
            ]},
            {'accuracy': 80.0, 'details': [
                {'pred': i, 'answer': i, 'correct': i % 2 == 1}
                for i in range(3)
            ]},
        ])
        results = self._run(evaluator, n=2, size=3)
        self.assertAlmostEqual(results['accuracy (2 runs average)'], 70.0)
        self.assertIn('details', results)
        self.assertEqual(len(results['details']), 3)

    def test_single_run_without_details_returns_raw(self):
        """n=1 without details: raw score() output is returned unchanged."""
        evaluator = ScriptedEvaluator([
            {'accuracy': 60.0, 'note': 'raw passthrough'},
        ])
        results = self._run(evaluator, n=1, size=3)
        self.assertEqual(results, {'accuracy': 60.0,
                                   'note': 'raw passthrough'})


if __name__ == '__main__':
    unittest.main()
