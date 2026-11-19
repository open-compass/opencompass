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


class TestGetLlmCorrectness(unittest.TestCase):
    """Regression for issue #2657.

    A judge reply that merely begins with the letters "CORRECT" — e.g.
    "Correctness: No, the answer is wrong." or "Correctly? No. B" — must not
    be counted as a positive verdict, while unambiguous positive verdicts
    such as "A", "CORRECT" or "correct." remain correct.
    """

    CASES = [
        # (llm_detail, expected, note)
        ({
            'prediction': 'A'
        }, True, 'positive letter A'),
        ({
            'prediction': 'CORRECT'
        }, True, 'positive word CORRECT'),
        ({
            'prediction': 'correct.'
        }, True, 'case-insensitive, trailing punct tolerated'),
        ({
            'prediction': 'B'
        }, False, 'negative letter B'),
        ({
            'prediction': 'INCORRECT'
        }, False, 'negative word'),
        # regression: previously miscounted as correct by startswith('CORRECT')
        ({
            'prediction': 'Correctness: No, the answer is wrong.'
        }, False, 'starts with "Correctness" but verdict is negative'),
        ({
            'llm_judge': 'Correctness: incorrect, does not match gold.'
        }, False, 'same defect via the llm_judge key'),
        ({
            'prediction':
            'Correctness is not met; the final numeric answer '
            'differs from the gold target.'
        }, False, 'any verdict opening with "Correctness"'),
        ({
            'prediction': 'Correctly? No. B'
        }, False, '"Correctly" also matched the old prefix'),
        ({
            'prediction': ''
        }, False, 'empty reply is not correct'),
        ({
            'correct': True
        }, True, 'passes through explicit correct flag'),
        ({
            'correct': False
        }, False, 'passes through explicit incorrect flag'),
        ({
            'score': 0.8
        }, True, 'score > 0.5 is correct'),
        ({
            'score': 0.2
        }, False, 'score <= 0.5 is not correct'),
        ({}, False, 'no recognised field is not correct'),
    ]

    def test_verdicts(self):
        ev = CascadeEvaluator.__new__(CascadeEvaluator)
        for detail, expected, note in self.CASES:
            with self.subTest(note=note):
                self.assertEqual(ev._get_llm_correctness(detail),
                                 expected,
                                 msg=note)


if __name__ == '__main__':
    unittest.main()
