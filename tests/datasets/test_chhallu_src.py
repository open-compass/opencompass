import json
import unittest

from opencompass.datasets.chhallu_src import CHHalluSrcEvaluator, _parse_json


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

    def test_answer_after_echoed_format_is_scored_correct(self):
        # A model that restates the required answer format and then gives its
        # real answer must still be scored on the real answer. A greedy
        # ``{.*}`` match fuses both braces into invalid JSON and scores the
        # correct answer as wrong.
        reference = {'type': 'attribution', 'answer': {'choice': 'C'}}
        prediction = ('题目要求以 {"choice": "A"} 格式作答。'
                      '经分析，正确选项是 C。\n{"choice": "C"}')
        result = self.evaluator.score([prediction], [reference])
        self.assertEqual(result['accuracy'], 100.0)

    def test_parse_json_returns_last_valid_object(self):
        self.assertEqual(_parse_json('{"choice": "A"}'), {'choice': 'A'})
        self.assertEqual(
            _parse_json('format: {"choice": "A"}\nanswer: {"choice": "C"}'),
            {'choice': 'C'})
        self.assertEqual(_parse_json('{"a": {"b": 1}}'), {'a': {'b': 1}})
        self.assertIsNone(_parse_json('no json here'))
        self.assertIsNone(_parse_json(''))
        self.assertIsNone(_parse_json(None))

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
