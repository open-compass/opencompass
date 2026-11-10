import unittest

from opencompass.evaluator.cascade_evaluator import CascadeEvaluator


class RuleEvaluatorWithoutTestSet:

    def __init__(self):
        self.calls = []

    def score(self, predictions, references):
        self.calls.append((predictions, references))
        return {
            'details': [{
                'pred': predictions[0],
                'answer': references[0],
                'correct': True,
            }]
        }


class RuleEvaluatorWithTestSet(RuleEvaluatorWithoutTestSet):

    def score(self, predictions, references, test_set=None):
        self.calls.append((predictions, references, test_set))
        return {
            'details': [{
                'pred': predictions[0],
                'answer': references[0],
                'correct': True,
            }]
        }


class TestCascadeEvaluator(unittest.TestCase):

    def _make_evaluator(self, rule_evaluator):
        evaluator = CascadeEvaluator.__new__(CascadeEvaluator)
        evaluator.sample_score_fn = None
        evaluator.rule_evaluator = rule_evaluator
        return evaluator

    def test_sample_score_without_test_set_argument(self):
        rule_evaluator = RuleEvaluatorWithoutTestSet()
        evaluator = self._make_evaluator(rule_evaluator)

        result = evaluator.sample_score('prediction', 'reference',
                                        {'input': 'question'})

        self.assertTrue(result['correct'])
        self.assertEqual(rule_evaluator.calls,
                         [(['prediction'], ['reference'])])

    def test_sample_score_with_test_set_argument(self):
        rule_evaluator = RuleEvaluatorWithTestSet()
        evaluator = self._make_evaluator(rule_evaluator)

        test_item = {'input': 'question'}
        result = evaluator.sample_score('prediction', 'reference', test_item)

        self.assertTrue(result['correct'])
        self.assertEqual(rule_evaluator.calls,
                         [(['prediction'], ['reference'], [test_item])])


if __name__ == '__main__':
    unittest.main()
