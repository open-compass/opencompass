import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from opencompass.datasets.squad20 import SQuAD20Dataset, SQuAD20Evaluator


class TestSQuAD20(unittest.TestCase):

    def setUp(self):
        qas = [
            dict(question='Who wrote the book?',
                 is_impossible=False,
                 answers=[dict(text='Jane Austen'),
                          dict(text='Austen')]),
            dict(question='Who illustrated the book?',
                 is_impossible=True,
                 answers=[],
                 plausible_answers=[dict(text='Jane Austen')]),
        ]
        paragraph = dict(context='Jane Austen wrote the book.', qas=qas)
        data = json.dumps({'data': [{'paragraphs': [paragraph]}]})
        with patch('opencompass.datasets.squad20.open',
                   mock_open(read_data=data)):
            self.rows = SQuAD20Dataset.load(str(Path('dev.json').resolve()))
        self.evaluator = SQuAD20Evaluator()

    def test_answerable_aliases(self):
        # Official evaluate-v2.0.py takes the best match over qa['answers'].
        refs = [self.rows[0]['answers']]
        for alias in ('Jane Austen', 'Austen'):
            with self.subTest(alias=alias):
                result = self.evaluator.score([alias], refs)
                self.assertEqual(result['score'], 100)

    def test_unanswerable(self):
        # With no gold answers the official script uses ''. The prompt uses
        # 'impossible to answer' for the same abstention.
        refs = [self.rows[1]['answers']]
        distractor = self.evaluator.score(['Jane Austen'], refs)
        self.assertEqual(distractor['score'], 0)
        self.assertEqual(
            self.evaluator.score(['impossible to answer'], refs)['score'], 100)
