"""Unit tests for BasePartitioner."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from mmengine.config import ConfigDict


try:
    from opencompass.partitioners.base import BasePartitioner
    BASE_PARTITIONER_AVAILABLE = True
except ImportError:
    BASE_PARTITIONER_AVAILABLE = False


class TestBasePartitioner(unittest.TestCase):
    """Test cases for BasePartitioner."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_initialization(self):
        """Test BasePartitioner initialization."""
        if not BASE_PARTITIONER_AVAILABLE:
            self.skipTest("BasePartitioner not available")
        
        partitioner = BasePartitioner(out_dir=self.temp_dir)
        self.assertEqual(partitioner.out_dir, self.temp_dir)
        self.assertIsNotNone(partitioner.keep_keys)

    def test_initialization_with_custom_keep_keys(self):
        """Test BasePartitioner initialization with custom keep_keys."""
        if not BASE_PARTITIONER_AVAILABLE:
            self.skipTest("BasePartitioner not available")
        
        custom_keys = ['custom.key1', 'custom.key2']
        partitioner = BasePartitioner(out_dir=self.temp_dir, keep_keys=custom_keys)
        self.assertEqual(partitioner.keep_keys, custom_keys)

    def test_parse_model_dataset_args(self):
        """Test parse_model_dataset_args method."""
        if not BASE_PARTITIONER_AVAILABLE:
            self.skipTest("BasePartitioner not available")
        
        partitioner = BasePartitioner(out_dir=self.temp_dir)
        
        cfg = ConfigDict({
            'models': [ConfigDict({'abbr': 'model1'})],
            'datasets': [ConfigDict({'abbr': 'dataset1'})]
        })
        
        result = partitioner.parse_model_dataset_args(cfg)
        self.assertIn('models', result)
        self.assertIn('datasets', result)
        self.assertEqual(len(result['models']), 1)
        self.assertEqual(len(result['datasets']), 1)

    def test_parse_model_dataset_args_with_combinations(self):
        """Test parse_model_dataset_args with model_dataset_combinations."""
        if not BASE_PARTITIONER_AVAILABLE:
            self.skipTest("BasePartitioner not available")
        
        # Create a partitioner that supports model_dataset_combinations
        # We'll use a mock to test the logic
        class MockPartitioner(BasePartitioner):
            def partition(self, model_dataset_combinations, **kwargs):
                return []
        
        partitioner = MockPartitioner(out_dir=self.temp_dir)
        
        cfg = ConfigDict({
            'models': [ConfigDict({'abbr': 'model1'})],
            'datasets': [ConfigDict({'abbr': 'dataset1'})],
            'model_dataset_combinations': [{
                'models': [ConfigDict({'abbr': 'model1'})],
                'datasets': [ConfigDict({'abbr': 'dataset1'})]
            }]
        })
        
        result = partitioner.parse_model_dataset_args(cfg)
        self.assertIn('model_dataset_combinations', result)


if __name__ == '__main__':
    unittest.main()
