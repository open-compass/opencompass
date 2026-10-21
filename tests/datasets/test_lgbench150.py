import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from datasets import Dataset

from opencompass.datasets.lgbench150 import (LGBench150Evaluator,
                                             _extract_option, _load_graph,
                                             _load_jsonl)
from opencompass.utils.datasets_info import DATASETS_MAPPING, DATASETS_URL


class LGBench150Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls.temp_dir.name)
        cls.questions = [
            dict(id=1,
                 category='medical',
                 question='medical one',
                 A='a',
                 B='b',
                 C='c',
                 D='d',
                 answer='A'),
            dict(id=2,
                 category='medical',
                 question='medical two',
                 A='a',
                 B='b',
                 C='c',
                 D='d',
                 answer='B'),
            dict(id=3,
                 category='biology',
                 question='biology one',
                 A='a',
                 B='b',
                 C='c',
                 D='d',
                 answer='C'),
            dict(id=4,
                 category='biology',
                 question='biology two',
                 A='a',
                 B='b',
                 C='c',
                 D='d',
                 answer='D'),
            dict(id=5,
                 category='chemistry',
                 question='chemistry one',
                 A='a',
                 B='b',
                 C='c',
                 D='d',
                 answer='A'),
            dict(id=6,
                 category='chemistry',
                 question='chemistry two',
                 A='a',
                 B='b',
                 C='c',
                 D='d',
                 answer='B'),
        ]
        cls.edges = [
            dict(source=1, target=2, weight=0.8),
            dict(source=3, target=4, weight=0.8),
            dict(source=5, target=6, weight=0.8),
        ]
        (cls.data_dir / 'questions.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in cls.questions),
            encoding='utf-8')
        (cls.data_dir / 'graph.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in cls.edges),
            encoding='utf-8')
        cls.test_set = Dataset.from_list(cls.questions)
        cls.references = [row['answer'] for row in cls.questions]
        cls.evaluator = LGBench150Evaluator(data_dir=cls.data_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_dataset_and_graph(self):
        self.assertEqual(len(_load_jsonl(self.evaluator.questions_path)), 6)
        self.assertEqual(len(_load_jsonl(self.evaluator.graph_path)), 3)
        counts = {
            domain: sum(row['category'] == domain for row in self.questions)
            for domain in ('medical', 'biology', 'chemistry')
        }
        self.assertEqual(counts, {
            'medical': 2,
            'biology': 2,
            'chemistry': 2
        })
        categories = np.asarray([row['category'] for row in self.questions])
        src, dst, weights = _load_graph(self.evaluator.graph_path, categories,
                                        0.4)
        self.assertEqual((len(src), len(dst), len(weights)), (3, 3, 3))

    def test_dataset_download_registration(self):
        self.assertEqual(DATASETS_MAPPING['opencompass/lgbench150']['local'],
                         './data/lgbench150_data')
        self.assertEqual(
            DATASETS_URL['/lgbench150_data']['md5'],
            '9013a5c35d2ad5ff1bf5960c26d290cc')

    def test_evaluator_resolves_data_dir(self):
        with patch('opencompass.datasets.lgbench150.get_data_path',
                   return_value='/cache/data/lgbench150_data') as resolver:
            evaluator = LGBench150Evaluator()

        resolver.assert_called_once_with('opencompass/lgbench150')
        self.assertEqual(evaluator.questions_path,
                         Path('/cache/data/lgbench150_data/questions.jsonl'))
        self.assertEqual(evaluator.graph_path,
                         Path('/cache/data/lgbench150_data/graph.jsonl'))

    def test_answer_extraction_is_strict(self):
        self.assertEqual(_extract_option('C'), 'C')
        self.assertEqual(_extract_option('Answer: (b).'), 'B')
        self.assertEqual(_extract_option('A or C'), '')
        self.assertEqual(_extract_option('The answer is C'), '')

    def test_perfect_and_zero_predictions(self):
        perfect = self.evaluator.score(self.references, self.references,
                                       self.test_set)
        for scope in ('overall', 'medical', 'biology', 'chemistry'):
            self.assertEqual(perfect[f'{scope}/Acc'], 100)
            self.assertEqual(perfect[f'{scope}/GCS'], 100)
            self.assertEqual(perfect[f'{scope}/KBS'], 0)

        zero = self.evaluator.score(['?'] * len(self.questions),
                                    self.references, self.test_set)
        for scope in ('overall', 'medical', 'biology', 'chemistry'):
            self.assertEqual(zero[f'{scope}/Acc'], 0)
            self.assertEqual(zero[f'{scope}/GCS'], 0)
            self.assertEqual(zero[f'{scope}/KBS'], 0)

    def test_rejects_misaligned_dataset(self):
        reversed_data = Dataset.from_list(list(reversed(self.questions)))
        with self.assertRaisesRegex(ValueError, 'order'):
            self.evaluator.score(self.references, self.references, reversed_data)


if __name__ == '__main__':
    unittest.main()
