import unittest
from tempfile import TemporaryDirectory

from datasets import Dataset

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


class StubLLMEvaluator:

    def __init__(self):
        self.calls = []

    def pred_postprocess(self, predictions):
        return predictions

    def score(self, predictions, references, test_set=None):
        self.calls.append((predictions, references, test_set.to_list()))
        return {'details': [{'correct': True} for _ in predictions]}


class StrippingRuleEvaluator(RuleEvaluatorWithoutTestSet):

    def pred_postprocess(self, predictions):
        return [prediction.strip() for prediction in predictions]


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

    def test_score_with_sample_score_fn_without_rule_evaluator(self):
        test_set = Dataset.from_dict({'question': ['first', 'second']})
        predictions = [' answer ', 'wrong']
        references = ['answer', 'answer']

        for parallel in (False, True):
            for return_dict in (False, True):
                with self.subTest(parallel=parallel, return_dict=return_dict):
                    calls = []

                    def sample_score(prediction, reference, test_item):
                        calls.append((prediction, reference, test_item))
                        correct = prediction.strip() == reference
                        return {'correct': correct} if return_dict else correct

                    evaluator = CascadeEvaluator(
                        llm_evaluator={'type': StubLLMEvaluator},
                        sample_score_fn=sample_score,
                        parallel=parallel,
                    )
                    with TemporaryDirectory() as tmp_dir:
                        evaluator._out_dir = f'{tmp_dir}/results'
                        result = evaluator.score(predictions, references,
                                                 test_set)

                    self.assertEqual(calls, [
                        (' answer ', 'answer', {
                            'question': 'first'
                        }),
                        ('wrong', 'answer', {
                            'question': 'second'
                        }),
                    ])
                    self.assertEqual(result['accuracy'], 100)
                    self.assertEqual(result['cascade_stats']['rule_correct'],
                                     1)
                    self.assertEqual(result['cascade_stats']['llm_evaluated'],
                                     2 if parallel else 1)
                    self.assertTrue(
                        all(item['cascade_correct']
                            for item in result['details']))
                    start = 0 if parallel else 1
                    self.assertEqual(
                        evaluator.llm_evaluator.calls,
                        [(predictions[start:], references[start:],
                          [{
                              'question': row['question'],
                              'prediction': pred,
                              'reference': ref,
                          }
                           for row, pred, ref in zip(
                               test_set.select(range(start, 2)),
                               predictions[start:], references[start:])])])

    def test_score_preserves_rule_prediction_postprocessing(self):
        evaluator = CascadeEvaluator(
            llm_evaluator={'type': StubLLMEvaluator},
            rule_evaluator={'type': StrippingRuleEvaluator},
            parallel=False,
        )

        result = evaluator.score([' prediction '], ['reference'])

        self.assertEqual(evaluator.rule_evaluator.calls,
                         [(['prediction'], ['reference'])])
        self.assertEqual(result['accuracy'], 100)
        self.assertEqual(evaluator.llm_evaluator.calls, [])


if __name__ == '__main__':
    unittest.main()
