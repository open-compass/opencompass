import json
import os
import tempfile
import unittest

from opencompass.datasets.infinitebench import (
    InfiniteBenchretrievekvDataset, InfiniteBenchretrievekvEvaluator,
    InfiniteBenchretrievenumberDataset, InfiniteBenchretrievepasskeyDataset)
from opencompass.datasets.infinitebench.utils import \
    InfiniteBench_first_number_postprocess
from opencompass.openicl.icl_evaluator import AccEvaluator


def _row(answer, input=''):
    # Same fields as the official InfiniteBench jsonl files; the long
    # context is not needed for scoring.
    return dict(id=0, context='', input=input, answer=answer, options=[])


class TestInfiniteBenchRetrievalAnswers(unittest.TestCase):
    """The official data stores the answer of the retrieval tasks as a
    one-element list, e.g. ``"answer": ["71432"]``."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _load(self, dataset_cls, rows):
        path = os.path.join(self.tmpdir.name, 'task.jsonl')
        with open(path, 'w') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
        return dataset_cls.load(path)

    def _acc(self, dataset, predictions):
        # Evaluation as configured in infinitebench_retrievepasskey_gen and
        # infinitebench_retrievenumber_gen.
        predictions = [
            InfiniteBench_first_number_postprocess(p) for p in predictions
        ]
        return AccEvaluator().score(predictions, dataset['answer'])['accuracy']

    def test_passkey(self):
        dataset = self._load(
            InfiniteBenchretrievepasskeyDataset,
            [_row(['71432']), _row(['69079'])])
        self.assertEqual(
            self._acc(dataset, ['The pass key is 71432.', '69079']), 100.0)
        self.assertEqual(
            self._acc(dataset, ['The pass key is 71432.', '12345']), 50.0)
        self.assertEqual(dataset['answer'], ['71432', '69079'])

    def test_number_string(self):
        dataset = self._load(InfiniteBenchretrievenumberDataset,
                             [_row(['3000077442'])])
        self.assertEqual(self._acc(dataset, ['The sequence is 3000077442.']),
                         100.0)
        self.assertEqual(dataset['answer'], ['3000077442'])

    def test_kv_retrieval(self):
        value = '5e6b7b90-710d-4953-9b18-3e96b2cadbf2'
        dataset = self._load(InfiniteBenchretrievekvDataset, [_row([value])])
        evaluator = InfiniteBenchretrievekvEvaluator()
        self.assertEqual(
            evaluator.score([f'The value is "{value}".'],
                            dataset['answer'])['score'], 100.0)
        self.assertEqual(
            evaluator.score(['The value is "something-else".'],
                            dataset['answer'])['score'], 0.0)
        self.assertEqual(dataset['answer'], [value])

    def test_plain_answer_is_unchanged(self):
        # kv_retrieval.jsonl stored a plain string from December 2023 until
        # its August 2024 update.
        value = '5e6b7b90-710d-4953-9b18-3e96b2cadbf2'
        dataset = self._load(InfiniteBenchretrievekvDataset, [_row(value)])
        self.assertEqual(dataset['answer'], [value])
        self.assertEqual(
            InfiniteBenchretrievekvEvaluator().score(
                [f'The value is "{value}".'], dataset['answer'])['score'],
            100.0)


if __name__ == '__main__':
    unittest.main()
