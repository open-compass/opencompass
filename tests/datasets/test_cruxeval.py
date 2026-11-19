import unittest

from opencompass.datasets.cruxeval import CRUXEvalOEvaluator


class TestCRUXEvalOEvaluator(unittest.TestCase):

    @staticmethod
    def score(prediction, output='1'):
        test_set = [{
            'id': 'sample',
            'code': 'def f(x):\n    return x',
            'input': '1',
            'output': output,
        }]
        return CRUXEvalOEvaluator().score([prediction], [output], test_set)

    def test_rejects_prediction_with_forged_equality(self):
        prediction = (
            "type('AlwaysEqual', (), "
            "{'__eq__': lambda self, other: True})()")

        result = self.score(prediction)

        self.assertEqual(result['CRUXEval-O'], 0.0)
        self.assertFalse(result['details']['0']['is_correct'])

    def test_rejects_nested_prediction_with_forged_equality(self):
        prediction = (
            "[type('AlwaysEqual', (), "
            "{'__eq__': lambda self, other: True})()]")

        result = self.score(prediction, output='[1]')

        self.assertEqual(result['CRUXEval-O'], 0.0)
        self.assertFalse(result['details']['0']['is_correct'])

    def test_prediction_cannot_change_outer_assertion(self):
        for prediction in ['0 or True', '0) or True #']:
            with self.subTest(prediction=prediction):
                result = self.score(prediction)

                self.assertEqual(result['CRUXEval-O'], 0.0)
                self.assertFalse(result['details']['0']['is_correct'])

    def test_accepts_matching_nested_builtin_in_multiple_predictions(self):
        prediction = ['None', "{'value': [1, (2, 3)]}"]

        result = self.score(prediction, output="{'value': [1, (2, 3)]}")

        self.assertEqual(result['CRUXEval-O'], 100.0)
        self.assertTrue(result['details']['0']['is_correct'])

    def test_preserves_echo_anti_cheat(self):
        result = self.score('f(1)')

        self.assertEqual(result['CRUXEval-O'], 0.0)
        self.assertFalse(result['details']['0']['is_correct'])


if __name__ == '__main__':
    unittest.main()
