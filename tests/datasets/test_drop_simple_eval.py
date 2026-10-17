import unittest

from opencompass.datasets.drop_simple_eval import (DropOpenAIEvaluator,
                                                   drop_metric)


class TestDropSimpleEval(unittest.TestCase):

    def test_drop_metric_normalizes_numbers(self):
        self.assertEqual(drop_metric('18.0', ['18']), (1.0, 100.0))

    def test_drop_metric_normalizes_text(self):
        cases = [
            ('The U.S.', 'US'),
            ('New\tYork', 'New York'),
            ('New-York', 'New York'),
        ]

        for prediction, reference in cases:
            with self.subTest(prediction=prediction, reference=reference):
                self.assertEqual(drop_metric(prediction, [reference]),
                                 (1.0, 100.0))

    def test_drop_metric_rejects_overlap_with_different_numbers(self):
        self.assertEqual(drop_metric('18 yards', ['19 yards']), (0.0, 0.0))

    def test_drop_metric_handles_empty_prediction_and_reference(self):
        self.assertEqual(drop_metric('', ['', 'answer']), (0.0, 0.0))

    def test_evaluator_reports_em_and_f1_over_alternative_answers(self):
        result = DropOpenAIEvaluator().score(
            predictions=['Answer: Denver', 'Answer: Broncos'],
            references=['Denver Broncos|Broncos', 'Denver Broncos|Broncos'],
        )

        self.assertEqual(result['accuracy'], 100.0)
        self.assertEqual(result['exact_match'], 50.0)
        self.assertAlmostEqual(result['f1'], 83.335)
        self.assertTrue(result['details'][0]['correct'])
        self.assertEqual(result['details'][0]['exact_match'], 0.0)
        self.assertEqual(result['details'][0]['f1'], 66.67)
        self.assertEqual(result['details'][1]['exact_match'], 1.0)
        self.assertEqual(result['details'][1]['f1'], 100.0)


if __name__ == '__main__':
    unittest.main()
