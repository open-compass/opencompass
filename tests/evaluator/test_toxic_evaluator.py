import unittest

from opencompass.openicl.icl_evaluator import ToxicEvaluator


class TestToxicEvaluator(unittest.TestCase):

    def _get_metrics(self, *scores):
        evaluator = ToxicEvaluator.__new__(ToxicEvaluator)
        evaluator.thr = 0.5
        score_dict = {
            str(index): {
                'toxicity_score': score
            }
            for index, score in enumerate(scores)
        }
        return evaluator.get_metrics(score_dict)

    def test_metrics_ignore_invalid_scores_when_finding_maximum(self):
        metrics = self._get_metrics(-1, 0.9, 0.1)

        self.assertEqual(metrics['expected_max_toxicity'], 0.9)
        self.assertEqual(metrics['valid_frac'], 0.6667)
        self.assertEqual(metrics['toxic_frac_valid'], 0.5)
        self.assertEqual(metrics['avg_toxicity_score'], 0.5)

    def test_maximum_is_independent_of_invalid_score_order(self):
        score_orders = [(-1, 0.1, 0.9), (0.1, -1, 0.9), (0.1, 0.9, -1)]

        for order in score_orders:
            with self.subTest(order=order):
                self.assertEqual(
                    self._get_metrics(*order)['expected_max_toxicity'], 0.9)

    def test_all_invalid_scores_raise_value_error(self):
        with self.assertRaisesRegex(ValueError,
                                    'All predictions are not valid'):
            self._get_metrics(-1, -1)


if __name__ == '__main__':
    unittest.main()
