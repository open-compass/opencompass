import unittest

from datasets import Dataset

from opencompass.openicl.icl_evaluator import SquadEvaluator
from opencompass.registry import ICL_EVALUATORS


class TestSquadEvaluator(unittest.TestCase):

    def test_score_local_metric(self):
        evaluator = ICL_EVALUATORS.build(dict(type='SquadEvaluator'))
        cases = [
            ('The Eiffel Tower.', 'Eiffel Tower', 100.0),
            ('London', 'Paris', 0.0),
            ('red', 'red blue green', 50.0),
            ('Paris\nAdditional explanation', 'Paris', 100.0),
        ]
        for prediction, reference, expected in cases:
            with self.subTest(prediction=prediction):
                self.assertEqual(evaluator.score([prediction], [reference]),
                                 {'f1': expected})

    def test_evaluate_single_replica(self):
        dataset = Dataset.from_list([{'question': 'Capital?'}] * 2)
        result = SquadEvaluator().evaluate(k=1,
                                           n=1,
                                           original_dataset=dataset,
                                           predictions=['Paris', 'London'],
                                           references=['Paris', 'Paris'])
        self.assertEqual(result, {'f1': 50.0})

    def test_evaluate_multiple_replicas(self):
        dataset = Dataset.from_list([{'question': 'Capital?'}] * 2)
        result = SquadEvaluator().evaluate(k=1,
                                           n=2,
                                           original_dataset=dataset,
                                           predictions=['Paris', 'London'],
                                           references=['Paris', 'Paris'])
        self.assertEqual(result, {'f1 (2 runs average)': 50.0})


if __name__ == '__main__':
    unittest.main()
