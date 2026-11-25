import unittest

from opencompass.datasets.infinitebench import InfiniteBenchmathcalcEvaluator


class TestInfiniteBenchMathCalcEvaluator(unittest.TestCase):
    """Expected scores follow ``get_score_one_math_calc`` in the official
    InfiniteBench ``src/compute_scores.py``: per example, the number of
    values that are right before the first mistake, divided by the number of
    expected values, averaged over examples."""

    def setUp(self):
        self.evaluator = InfiniteBenchmathcalcEvaluator()

    def _score(self, predictions, references):
        return self.evaluator.score(predictions, references)['score']

    def test_all_values_right(self):
        # The in-context example of the prompt: 8 - 3 + 2 - 4
        self.assertAlmostEqual(self._score(['[5, 7, 3]'], [[5, 7, 3]]), 100.0)

    def test_partially_right(self):
        self.assertAlmostEqual(self._score(['[5, 7, 4]'], [[5, 7, 3]]),
                               200 / 3)
        self.assertAlmostEqual(self._score(['[5]'], [[5, 7, 3]]), 100 / 3)

    def test_counting_stops_at_first_mistake(self):
        self.assertAlmostEqual(self._score(['[6, 7, 3]'], [[5, 7, 3]]), 0.0)

    def test_average_over_examples(self):
        self.assertAlmostEqual(
            self._score(['[5, 7, 3]', '[4]'], [[5, 7, 3], [4, 8]]), 75.0)

    def test_long_expression(self):
        # The official math_calc.jsonl has 14999 values per example.
        reference = list(range(14999))
        prediction = str(reference[:10])
        self.assertAlmostEqual(self._score([prediction], [reference]),
                               10 / 14999 * 100)
        self.assertAlmostEqual(self._score([str(reference)], [reference]),
                               100.0)

    def test_nested_reference(self):
        # math_calc.jsonl before its August 2024 update: [[v1, v2, ...]]
        self.assertAlmostEqual(self._score(['[5, 7, 4]'], [[[5, 7, 3]]]),
                               200 / 3)


if __name__ == '__main__':
    unittest.main()
