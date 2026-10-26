import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opencompass.evaluator.math_evaluator import MATHVerifyEvaluator


def fake_verify(prediction, reference, timeout=10):
    return {
        'answer_correct': float(prediction == reference),
        'answer_parsed': prediction,
        'gold_parsed': reference,
    }


def fake_verify_with_skipped(prediction, reference, timeout=10):
    if prediction == 'skip':
        return {'skipped': True}
    return fake_verify(prediction, reference, timeout)


class TestMathVerifyEvaluator(unittest.TestCase):

    def _mock_math_verify_deps(self):
        return patch.dict(
            sys.modules,
            {
                'latex2sympy2_extended':
                SimpleNamespace(NormalizationConfig=object),
                'math_verify':
                SimpleNamespace(
                    ExprExtractionConfig=object,
                    LatexExtractionConfig=object,
                    parse=object,
                    verify=object,
                ),
            },
        )

    def test_keeps_string_prediction_behavior(self):
        with self._mock_math_verify_deps(), patch(
                'opencompass.evaluator.math_evaluator._verify_with_timeout',
                side_effect=fake_verify):
            result = MATHVerifyEvaluator().score(['4'], ['4'])

        self.assertEqual(result['accuracy'], 100.0)
        self.assertEqual(result['details'], [{
            'pred': '4',
            'answer': '4',
            'correct': True,
        }])

    def test_scores_multi_predictions_as_pass_at_1(self):
        with self._mock_math_verify_deps(), patch(
                'opencompass.evaluator.math_evaluator._verify_with_timeout',
                side_effect=fake_verify):
            result = MATHVerifyEvaluator().score([['4', '5']], ['4'])

        self.assertEqual(result['accuracy'], 50.0)
        self.assertEqual(result['details'][0]['pass_at_1'], 0.5)
        self.assertIs(result['details'][0]['correct'], False)
        self.assertEqual(result['details'][0]['candidate_details'], [
            {
                'pred': '4',
                'answer': '4',
                'correct': True,
            },
            {
                'pred': '5',
                'answer': '4',
                'correct': False,
            },
        ])

    def test_multi_prediction_details_match_scores_when_skipped(self):
        with self._mock_math_verify_deps(), patch(
                'opencompass.evaluator.math_evaluator._verify_with_timeout',
                side_effect=fake_verify_with_skipped):
            result = MATHVerifyEvaluator().score([['4', 'skip']], ['4'])

        detail = result['details'][0]
        self.assertEqual(result['accuracy'], 50.0)
        self.assertEqual(detail['pass_at_1'], 0.5)
        self.assertEqual(len(detail['candidate_details']), 2)
        self.assertEqual(detail['candidate_details'][1], {
            'pred': 'skip',
            'answer': '4',
            'correct': False,
            'skipped': True,
        })


if __name__ == '__main__':
    unittest.main()
