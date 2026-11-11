import unittest

from opencompass.datasets.QuALITY import QuALITYEvaluator


class TestQuALITYEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = QuALITYEvaluator()

    def test_scores_easy_and_hard_subsets_independently(self):
        result = self.evaluator.score(
            predictions=['A', 'B', 'C'],
            references=['A', 'B', 'C'],
            test_set=[
                {
                    'difficult': 0
                },
                {
                    'difficult': 0
                },
                {
                    'difficult': 1
                },
            ],
        )

        self.assertEqual(result, {
            'easy_acc': 100.0,
            'hard_acc': 100.0,
            'all_acc': 100.0,
        })

    def test_hard_only_subset(self):
        result = self.evaluator.score(
            predictions=['A', 'B'],
            references=['A', 'A'],
            test_set=[{
                'difficult': 1
            }, {
                'difficult': 1
            }],
        )

        self.assertEqual(result, {
            'easy_acc': 0.0,
            'hard_acc': 50.0,
            'all_acc': 50.0,
        })

    def test_easy_only_subset(self):
        result = self.evaluator.score(
            predictions=['A', 'B'],
            references=['A', 'A'],
            test_set=[{
                'difficult': 0
            }, {
                'difficult': 0
            }],
        )

        self.assertEqual(result, {
            'easy_acc': 50.0,
            'hard_acc': 0.0,
            'all_acc': 50.0,
        })

    def test_empty_input(self):
        result = self.evaluator.score([], [], [])

        self.assertEqual(result, {
            'easy_acc': 0.0,
            'hard_acc': 0.0,
            'all_acc': 0.0,
        })


if __name__ == '__main__':
    unittest.main()
