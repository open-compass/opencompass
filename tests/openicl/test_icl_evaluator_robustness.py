"""Unit tests for the evaluator robustness facilities (PR-1, G1).

Covers: per-sample isolation (protected_sample), timeout, failure
recording/export, max_failure_ratio batch guard, and the JudgeEvaluator
use case where unparseable judge responses are excluded from the
accuracy denominator.
"""

import time
import unittest

import datasets as hf_datasets

from opencompass.openicl.icl_evaluator.icl_base_evaluator import (
    BaseEvaluator,
)
from opencompass.openicl.icl_evaluator.icl_judge_evaluator import (
    JudgeEvaluator,
)


class RobustDummyEvaluator(BaseEvaluator):
    """Batch evaluator that isolates each sample via protected_sample."""

    def score(self, predictions, references):
        correct = 0
        count = 0
        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            outcome = self.protected_sample(self._check, idx, pred, ref)
            if 'error' in outcome:
                continue
            count += 1
            correct += int(outcome['correct'])
        if count == 0:
            return {'error': 'all samples failed'}
        return {'accuracy': 100 * correct / count}

    @staticmethod
    def _check(pred, ref):
        if ref == 'boom':
            raise RuntimeError('judge exploded')
        return {'correct': ref == pred}


class SleepyEvaluator(BaseEvaluator):
    """Single-sample evaluator whose sample check sleeps past timeout."""

    def score(self, predictions, references):
        outcome = self.protected_sample(self._slow, 0, predictions[0])
        return {'value': 0.0 if 'error' in outcome else 1.0}

    @staticmethod
    def _slow(pred):
        time.sleep(2.0)
        return pred


def _make_dataset(texts):
    return hf_datasets.Dataset.from_dict({
        'input': texts,
        'subdivision': ['test'] * len(texts),
        'idx': list(range(len(texts))),
    })


class TestProtectedSample(unittest.TestCase):

    def test_exception_recorded_not_raised(self):
        """Failure #1: exception is recorded and converted to error dict."""
        evaluator = RobustDummyEvaluator()
        result = evaluator.score(['a', 'b'], ['a', 'boom'])

        self.assertEqual(result['accuracy'], 100.0)
        self.assertEqual(len(evaluator._failures), 1)
        self.assertIn('RuntimeError', evaluator._failures[0]['error'])
        self.assertEqual(evaluator._failures[0]['idx'], 1)

    def test_timeout_recorded_early(self):
        """Failure #2: timed-out sample is recorded near the deadline."""
        evaluator = SleepyEvaluator(timeout=0.5)
        start = time.time()
        result = evaluator.score(['x'], ['x'])
        elapsed = time.time() - start

        self.assertEqual(result['value'], 0.0)
        self.assertLess(elapsed, 1.5)
        self.assertEqual(len(evaluator._failures), 1)
        self.assertIn('TimeoutError', evaluator._failures[0]['error'])

    def test_failure_mode_raise_propagates(self):
        evaluator = RobustDummyEvaluator(failure_mode='raise')

        with self.assertRaises(RuntimeError):
            evaluator.score(['a'], ['boom'])

    def test_export_failures_structured(self):
        evaluator = RobustDummyEvaluator()
        evaluator.score(['a', 'b', 'c'], ['a', 'boom', 'c'])

        exported = evaluator.export_failures()
        self.assertIsInstance(exported, list)
        self.assertEqual(len(exported), 1)
        for key in ('idx', 'error', 'elapsed'):
            self.assertIn(key, exported[0])


class TestEvaluateRobustness(unittest.TestCase):

    def test_max_failure_ratio_marks_batch_error(self):
        """Failure #3: exceeding the ratio fails the batch, not the score."""
        evaluator = RobustDummyEvaluator(max_failure_ratio=0.4)
        result = evaluator.evaluate(
            k=1,
            n=1,
            original_dataset=_make_dataset(['a', 'b']),
            predictions=['a', 'b'],
            references=['a', 'boom'],
        )

        self.assertIn('error', result)
        self.assertNotIn('accuracy', result)

    def test_default_behavior_unchanged(self):
        """Lock: with default init params, output has no robustness keys."""
        evaluator = RobustDummyEvaluator()
        result = evaluator.evaluate(
            k=1,
            n=1,
            original_dataset=_make_dataset(['a']),
            predictions=['a'],
            references=['a'],
        )

        self.assertEqual(result['accuracy'], 100.0)
        self.assertNotIn('failure_stats', result)
        self.assertNotIn('error', result)


class TestJudgeEvaluatorRobustness(unittest.TestCase):

    def test_unparseable_predictions_excluded_from_denominator(self):
        """Failure #6: unparseable judge responses are failures, not
        wrong answers, and are reported via failure_stats."""
        evaluator = JudgeEvaluator()
        predictions = [
            'answer is ... "Choice": "Model A',
            'verdict ... "Choice": "Model A',
            'I cannot decide between the models',
        ]
        references = [
            {'winner': 'A'},
            {'winner': 'B'},
            {'winner': 'A'},
        ]
        result = evaluator.evaluate(
            k=1,
            n=1,
            original_dataset=_make_dataset(['q1', 'q2', 'q3']),
            predictions=predictions,
            references=references,
        )

        self.assertEqual(result['accuracy'], 50.0)
        self.assertEqual(result['failure_stats']['count'], 1)
        self.assertAlmostEqual(result['failure_stats']['ratio'], 1 / 3)
        self.assertIn('error', result['details'][2])


if __name__ == '__main__':
    unittest.main()
