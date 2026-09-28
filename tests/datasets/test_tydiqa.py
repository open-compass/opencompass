import unittest

from opencompass.datasets.tydiqa import TydiQAEvaluator


class TestTydiQAEvaluator(unittest.TestCase):

    def test_score_ignores_case_of_references(self):
        evaluator = TydiQAEvaluator()
        result = evaluator.score(
            predictions=['Paris', 'Barack Obama was president'],
            references=[['Paris'], ['Barack Obama']])
        self.assertEqual(result['exact_match'], 50.0)
        self.assertAlmostEqual(result['f1'], (100.0 + 200 / 3) / 2)

    def test_score_wrong_answer(self):
        evaluator = TydiQAEvaluator()
        result = evaluator.score(predictions=['London'],
                                 references=[['Paris']])
        self.assertEqual(result, {'exact_match': 0.0, 'f1': 0.0})


if __name__ == '__main__':
    unittest.main()
