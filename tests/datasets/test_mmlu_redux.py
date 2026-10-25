"""Unit tests for the MMLU-Redux dataset loader."""

import unittest
from unittest.mock import patch

from datasets import Dataset

from opencompass.datasets.mmlu_redux import MMLUReduxDataset


class TestMMLUReduxDataset(unittest.TestCase):

    @staticmethod
    def _raw_dataset():
        return Dataset.from_dict({
            'question': ['q1', 'q2'],
            'choices': [
                ['w1', 'x1', 'y1', 'z1'],
                ['w2', 'x2', 'y2', 'z2'],
            ],
            'answer': [0, 3],
        })

    def test_load_expands_choices_and_target(self):
        with patch(
                'opencompass.datasets.mmlu_redux.load_dataset',
                return_value=self._raw_dataset(),
        ) as load:
            data = MMLUReduxDataset.load('fxmarty/mmlu-redux-2.0-ok',
                                         'abstract_algebra')

        load.assert_called_once_with('fxmarty/mmlu-redux-2.0-ok',
                                     'abstract_algebra',
                                     split='test')
        self.assertEqual(data[0]['A'], 'w1')
        self.assertEqual(data[0]['B'], 'x1')
        self.assertEqual(data[0]['C'], 'y1')
        self.assertEqual(data[0]['D'], 'z1')
        self.assertEqual(data[0]['target'], 'A')
        self.assertEqual(data[1]['target'], 'D')

    def test_load_forwards_extra_kwargs(self):
        with patch(
                'opencompass.datasets.mmlu_redux.load_dataset',
                return_value=self._raw_dataset(),
        ) as load:
            MMLUReduxDataset.load('some/path',
                                  'subject',
                                  trust_remote_code=True)

        load.assert_called_once_with('some/path',
                                     'subject',
                                     split='test',
                                     trust_remote_code=True)


if __name__ == '__main__':
    unittest.main()
