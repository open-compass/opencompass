"""Unit tests for NaivePartitioner."""

import tempfile
import unittest
from unittest.mock import patch

from mmengine.config import ConfigDict

from opencompass.partitioners.naive import NaivePartitioner


class TestNaivePartitioner(unittest.TestCase):
    """Test cases for NaivePartitioner."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_cfg = ConfigDict({
            'abbr': 'test_model',
            'type': 'TestModel',
            'path': 'test_path'
        })
        self.dataset_cfg = ConfigDict({
            'abbr': 'test_dataset',
            'type': 'TestDataset',
            'path': 'test_path'
        })

    def test_initialization(self):
        """Test NaivePartitioner initialization."""
        partitioner = NaivePartitioner(out_dir=self.temp_dir, n=1)
        self.assertEqual(partitioner.out_dir, self.temp_dir)
        self.assertEqual(partitioner.n, 1)

    def test_initialization_with_custom_n(self):
        """Test NaivePartitioner initialization with custom n."""
        partitioner = NaivePartitioner(out_dir=self.temp_dir, n=5)
        self.assertEqual(partitioner.n, 5)

    @patch('os.path.exists')
    def test_partition_creates_tasks(self, mock_exists):
        """Test that partition method creates tasks correctly."""
        mock_exists.return_value = False
        partitioner = NaivePartitioner(out_dir=self.temp_dir, n=1)

        model_dataset_combinations = [{
            'models': [self.model_cfg],
            'datasets': [self.dataset_cfg]
        }]

        tasks = partitioner.partition(
            model_dataset_combinations=model_dataset_combinations,
            work_dir=self.temp_dir,
            out_dir=self.temp_dir,
            add_cfg={})

        self.assertIsInstance(tasks, list)
        self.assertGreater(len(tasks), 0)
        self.assertIn('models', tasks[0])
        self.assertIn('datasets', tasks[0])
        self.assertIn('work_dir', tasks[0])

    @patch('os.path.exists')
    def test_partition_with_n_greater_than_one(self, mock_exists):
        """Test partition with n > 1 groups datasets."""
        mock_exists.return_value = False
        partitioner = NaivePartitioner(out_dir=self.temp_dir, n=2)

        # Create multiple datasets
        datasets = [
            ConfigDict({
                'abbr': f'test_dataset_{i}',
                'type': 'TestDataset',
                'path': 'test_path'
            }) for i in range(5)
        ]

        model_dataset_combinations = [{
            'models': [self.model_cfg],
            'datasets': datasets
        }]

        tasks = partitioner.partition(
            model_dataset_combinations=model_dataset_combinations,
            work_dir=self.temp_dir,
            out_dir=self.temp_dir,
            add_cfg={})

        # Should create tasks with at most n datasets each
        self.assertIsInstance(tasks, list)
        for task in tasks:
            self.assertLessEqual(len(task['datasets'][0]), partitioner.n)

    @patch('os.path.exists')
    def test_partition_skips_existing_files(self, mock_exists):
        """Test that partition skips tasks with existing output files."""
        # First call returns True (file exists), second returns False
        mock_exists.side_effect = [True, False]
        partitioner = NaivePartitioner(out_dir=self.temp_dir, n=1)

        datasets = [
            ConfigDict({
                'abbr': 'dataset1',
                'type': 'TestDataset',
                'path': 'test_path'
            }),
            ConfigDict({
                'abbr': 'dataset2',
                'type': 'TestDataset',
                'path': 'test_path'
            })
        ]

        model_dataset_combinations = [{
            'models': [self.model_cfg],
            'datasets': datasets
        }]

        tasks = partitioner.partition(
            model_dataset_combinations=model_dataset_combinations,
            work_dir=self.temp_dir,
            out_dir=self.temp_dir,
            add_cfg={})

        # Should only create task for dataset2 (dataset1 output exists)
        self.assertEqual(len(tasks), 1)

    def test_partition_with_add_cfg(self):
        """Test that partition includes add_cfg in tasks."""
        with patch('os.path.exists', return_value=False):
            partitioner = NaivePartitioner(out_dir=self.temp_dir, n=1)

            add_cfg = {'custom_key': 'custom_value'}
            model_dataset_combinations = [{
                'models': [self.model_cfg],
                'datasets': [self.dataset_cfg]
            }]

            tasks = partitioner.partition(
                model_dataset_combinations=model_dataset_combinations,
                work_dir=self.temp_dir,
                out_dir=self.temp_dir,
                add_cfg=add_cfg)

            self.assertIn('custom_key', tasks[0])
            self.assertEqual(tasks[0]['custom_key'], 'custom_value')


if __name__ == '__main__':
    unittest.main()
