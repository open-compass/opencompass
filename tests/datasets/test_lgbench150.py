import unittest

import numpy as np
from datasets import Dataset

from opencompass.datasets.lgbench150 import (GRAPH_FILE, LGBench150Evaluator,
                                             QUESTION_FILE, _extract_option,
                                             _load_graph, _load_jsonl)


class LGBench150Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.questions = _load_jsonl(QUESTION_FILE)
        cls.test_set = Dataset.from_list(cls.questions)
        cls.references = [row['answer'] for row in cls.questions]
        cls.evaluator = LGBench150Evaluator()

    def test_bundled_subset_and_graph(self):
        self.assertEqual(len(self.questions), 150)
        self.assertEqual(len(_load_jsonl(GRAPH_FILE)), 3436)
        counts = {
            domain: sum(row['category'] == domain for row in self.questions)
            for domain in ('medical', 'biology', 'chemistry')
        }
        self.assertEqual(counts, {
            'medical': 50,
            'biology': 50,
            'chemistry': 50
        })
        categories = np.asarray([row['category'] for row in self.questions])
        src, dst, weights = _load_graph(GRAPH_FILE, categories, 0.4)
        self.assertEqual((len(src), len(dst), len(weights)), (3436, 3436, 3436))

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
