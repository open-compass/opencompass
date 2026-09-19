import unittest

from opencompass.datasets.subjective.wildbench import (
    WildBenchDataset, WildBenchWithRawPromptDataset)


class TestWildBenchDataset(unittest.TestCase):

    def test_invalid_eval_mode(self):
        for dataset_cls in (WildBenchDataset, WildBenchWithRawPromptDataset):
            with self.subTest(dataset_cls=dataset_cls.__name__):
                with self.assertRaisesRegex(ValueError,
                                            'must be either "single" or "pair"'):
                    dataset_cls.load(None, 'unused', eval_mode='invalid')


if __name__ == '__main__':
    unittest.main()
