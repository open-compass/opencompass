"""Unit tests for SizePartitioner."""

import tempfile
import unittest

from mmengine.config import ConfigDict

from opencompass.partitioners.size import SizePartitioner


class TestSizePartitioner(unittest.TestCase):
    """Test cases for SizePartitioner."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.partitioner = SizePartitioner(
            out_dir=self.temp_dir.name,
            dataset_size_path=f'{self.temp_dir.name}/dataset_size.json')
        self.partitioner._dataset_size = {'demo': 1000}

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def _make_dataset_cfg(self, test_range):
        return ConfigDict({
            'abbr': 'demo',
            'reader_cfg': {
                'test_range': test_range
            },
            'infer_cfg': {
                'prompt_template': {
                    'template': '{question}'
                }
            }
        })

    def test_get_actual_size_handles_supported_test_range_types(self):
        """Test actual size calculation for supported test_range types."""
        cases = [
            (None, 1000),
            ('', 1000),
            (0, 1000),
            (-1, 1000),
            (1000, 1000),
            (100, 100),
            (0.5, 500),
            ('[:100]', 100),
            ('[100:200]', 100),
            ('[::2]', 500),
            ('[:100][10:20]', 10),
        ]

        for test_range, expected_size in cases:
            with self.subTest(test_range=test_range):
                self.assertEqual(
                    self.partitioner._get_actual_size(test_range, 1000),
                    expected_size)

    def test_get_cost_handles_non_string_test_range(self):
        """Test get_cost does not fail on non-string test_range values."""
        cases = [
            (None, (1000, 20), 20000),
            (100, (100, 20), 2000),
            (0.5, (500, 20), 10000),
            (0, (1000, 20), 20000),
        ]

        for test_range, raw_factors, expected_cost in cases:
            dataset_cfg = self._make_dataset_cfg(test_range)
            with self.subTest(test_range=test_range):
                self.assertEqual(
                    self.partitioner.get_cost(dataset_cfg,
                                              get_raw_factors=True),
                    raw_factors)
                self.assertEqual(self.partitioner.get_cost(dataset_cfg),
                                 expected_cost)


if __name__ == '__main__':
    unittest.main()
