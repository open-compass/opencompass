import json
import unittest

from opencompass.datasets.chhallu_src import CHHalluSrcEvaluator


class TestCHHalluSrcEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = CHHalluSrcEvaluator()
        self.crossbook_reference = {
            'type': 'crossbook',
            'answer': {'books': ['史记']},
            'grounding': {'era_window': [-200, -100]},
        }

    def test_invalid_book_elements_do_not_abort_batch(self):
        references = [
            {'type': 'attribution', 'answer': {'choice': 'B'}},
            self.crossbook_reference,
            {'type': 'sourcing', 'answer': {'chapters': ['淮阴侯列传']}},
        ]
        for invalid_book in (123, True, None, {}, [], 0):
            with self.subTest(invalid_book=invalid_book):
                predictions = [
                    json.dumps({'choice': 'B'}),
                    json.dumps({'books': ['史记', invalid_book]}),
                    json.dumps({'chapter': '淮阴侯列传'}),
                ]
                result = self.evaluator.score(predictions, references)
                self.assertAlmostEqual(result['accuracy'], 200 / 3)

    def test_valid_book_lists_keep_scoring_rules(self):
        cases = [
            (['史记'], 100.0),
            (['史記'], 100.0),
            (['史记', '汉书'], 100.0),
            ([], 0.0),
            (['汉书'], 0.0),
            (['史记', '后汉书'], 0.0),
        ]
        for books, accuracy in cases:
            with self.subTest(books=books):
                result = self.evaluator.score(
                    [json.dumps({'books': books})], [self.crossbook_reference])
                self.assertEqual(result['accuracy'], accuracy)


if __name__ == '__main__':
    unittest.main()
