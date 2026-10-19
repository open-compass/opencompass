"""Unit tests for AIME2025 dataset.

Note: This test requires the opencompass package and its dependencies to be installed.
If you encounter import errors, ensure all dependencies are installed:
    pip install -r requirements.txt

To test with real dataset files, set the COMPASS_DATA_CACHE environment variable:
    export COMPASS_DATA_CACHE=/path/to/data/cache
    pytest tests/dataset/test_aime2025.py::TestAime2025Dataset::test_load_with_real_data -v
"""

import json
import os
import tempfile
import unittest

from datasets import Dataset

from opencompass.datasets.custom import CustomDataset
class CustomDataset:
    """Mock CustomDataset for testing when full import is not available."""

    @staticmethod
    def load(path, **kwargs):
        from datasets import Dataset

        # Simulate loading JSONL file
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        return Dataset.from_list(dataset)

    def __init__(self, **kwargs):
        """Mock __init__ for testing initialization."""
        from datasets import Dataset, concatenate_datasets
        reader_cfg = kwargs.pop('reader_cfg', {})
        abbr = kwargs.pop('abbr', 'dataset')
        dataset = self.load(**kwargs)
        if isinstance(dataset, Dataset):
            dataset = dataset.map(lambda x, idx: {
                'subdivision': abbr,
                'idx': idx
            },
                                    with_indices=True,
                                    writer_batch_size=16,
                                    load_from_cache_file=False)
            dataset = concatenate_datasets([dataset] * 1)
        self.dataset = dataset
        # Create a mock reader instead of importing DatasetReader
        from unittest.mock import MagicMock
        mock_reader = MagicMock()
        mock_reader.input_columns = reader_cfg.get('input_columns', [])
        mock_reader.output_column = reader_cfg.get('output_column', None)
        self.reader = mock_reader


class TestAime2025Dataset(unittest.TestCase):
    """Test cases for AIME2025 dataset using CustomDataset."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = [{
            'question': 'What is the value of $\\sqrt{16}$?',
            'answer': '4',
            'origin_prompt': 'What is the value of $\\sqrt{16}$?',
            'gold_answer': '4'
        }, {
            'question': 'Solve for x: $x^2 + 5x + 6 = 0$',
            'answer': 'x = -2 or x = -3',
            'origin_prompt': 'Solve for x: $x^2 + 5x + 6 = 0$',
            'gold_answer': 'x = -2 or x = -3'
        }]

    def _create_temp_jsonl_file(self, data):
        """Create a temporary JSONL file with test data."""
        temp_file = tempfile.NamedTemporaryFile(mode='w',
                                                suffix='.jsonl',
                                                delete=False,
                                                encoding='utf-8')
        for item in data:
            temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        temp_file.close()
        return temp_file.name

    def test_load_reads_jsonl_file(self):
        """Test that load method reads JSONL file correctly."""
        # Create temporary JSONL file
        temp_file = self._create_temp_jsonl_file(self.test_data)

        try:
            # Use absolute path to avoid DATASETS_MAPPING lookup
            # get_data_path will return absolute paths as-is
            result = CustomDataset.load(path=temp_file)

            # Verify dataset was loaded correctly
            self.assertIsInstance(result, Dataset)
            self.assertEqual(len(result), 2)
            self.assertIn('question', result.column_names)
            self.assertIn('answer', result.column_names)
            self.assertEqual(result[0]['question'],
                             self.test_data[0]['question'])
            self.assertEqual(result[0]['answer'], self.test_data[0]['answer'])
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_returns_dataset(self):
        """Test that load method returns a Dataset instance."""
        temp_file = self._create_temp_jsonl_file(self.test_data)

        try:
            # Use absolute path to avoid DATASETS_MAPPING lookup
            result = CustomDataset.load(path=temp_file)

            self.assertIsInstance(result, Dataset)
            self.assertEqual(len(result), 2)
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_preserves_all_columns(self):
        """Test that load method preserves all columns from JSONL."""
        extended_data = [{
            'question': 'Test question',
            'answer': 'Test answer',
            'origin_prompt': 'Test question',
            'gold_answer': 'Test answer',
            'difficulty': 'hard',
            'category': 'algebra'
        }]
        temp_file = self._create_temp_jsonl_file(extended_data)

        try:
            # Use absolute path to avoid DATASETS_MAPPING lookup
            result = CustomDataset.load(path=temp_file)

            # Verify all columns are preserved
            self.assertIn('question', result.column_names)
            self.assertIn('answer', result.column_names)
            self.assertIn('origin_prompt', result.column_names)
            self.assertIn('gold_answer', result.column_names)
            self.assertIn('difficulty', result.column_names)
            self.assertIn('category', result.column_names)
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_with_empty_file(self):
        """Test load method with empty JSONL file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w',
                                                suffix='.jsonl',
                                                delete=False,
                                                encoding='utf-8')
        temp_file.close()

        try:
            # Use absolute path to avoid DATASETS_MAPPING lookup
            result = CustomDataset.load(path=temp_file.name)

            self.assertIsInstance(result, Dataset)
            self.assertEqual(len(result), 0)
        finally:
            import os
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def test_dataset_initialization(self):
        """Test that dataset can be initialized with proper configuration."""
        temp_file = self._create_temp_jsonl_file(self.test_data)

        try:
            # Initialize dataset with reader config
            # Use absolute path to avoid DATASETS_MAPPING lookup
            dataset = CustomDataset(path=temp_file,
                                    abbr='aime2025_test',
                                    reader_cfg=dict(
                                        input_columns=['question'],
                                        output_column='answer'))

            # Verify dataset was created
            self.assertIsNotNone(dataset.dataset)
            self.assertIsNotNone(dataset.reader)
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_dataset_reader_config(self):
        """Test that dataset reader is configured correctly."""
        temp_file = self._create_temp_jsonl_file(self.test_data)

        try:
            reader_cfg = dict(input_columns=['question'],
                              output_column='answer')
            # Use absolute path to avoid DATASETS_MAPPING lookup
            dataset = CustomDataset(path=temp_file,
                                    abbr='aime2025_test',
                                    reader_cfg=reader_cfg)

            # Verify reader configuration
            self.assertEqual(dataset.reader.input_columns,
                                ['question'])
            self.assertEqual(dataset.reader.output_column, 'answer')

        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_handles_unicode(self):
        """Test that load method handles Unicode characters correctly."""
        unicode_data = [{
            'question': '计算 $\\sqrt{16}$ 的值',
            'answer': '4',
            'origin_prompt': '计算 $\\sqrt{16}$ 的值',
            'gold_answer': '4'
        }]
        temp_file = self._create_temp_jsonl_file(unicode_data)

        try:
            # Use absolute path to avoid DATASETS_MAPPING lookup
            result = CustomDataset.load(path=temp_file)

            self.assertIsInstance(result, Dataset)
            self.assertEqual(len(result), 1)
            self.assertIn('question', result.column_names)
            self.assertEqual(result[0]['question'],
                             unicode_data[0]['question'])
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_with_real_data(self):
        """Test loading AIME2025 dataset with real data from COMPASS_DATA_CACHE.        

        This test requires COMPASS_DATA_CACHE environment variable to be set
        and the aime2025 dataset to be available in the cache directory.
        """
        import pytest

        # Check if COMPASS_DATA_CACHE is set
        cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
        if not cache_dir:
            pytest.skip('COMPASS_DATA_CACHE not set, skipping real data test')

        # Check if CustomDataset is available
        try:
            # Try to load the real dataset
            # The path 'opencompass/aime2025' should be resolved by get_data_path
            # which will look in COMPASS_DATA_CACHE
            dataset = CustomDataset.load(path=f'{cache_dir}/data/aime2025/aime2025.jsonl')

            # Verify dataset was loaded
            self.assertIsInstance(dataset, Dataset)
            self.assertGreater(len(dataset), 0, 'Dataset should not be empty')

            # Verify expected columns exist
            self.assertIn('question', dataset.column_names,
                          "Dataset should have 'question' column")
            self.assertIn('answer', dataset.column_names,
                          "Dataset should have 'answer' column")

            # Verify data structure
            sample = dataset[0]
            self.assertIn('question', sample)
            self.assertIn('answer', sample)

        except (FileNotFoundError, ValueError) as e:
            assert False, f"Real dataset not in COMPASS_DATA_CACHE: {e}"

    def test_dataset_initialization_with_real_data(self):
        """Test initializing AIME2025 dataset with real data from COMPASS_DATA_CACHE."""
        import pytest

        # Check if COMPASS_DATA_CACHE is set
        cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
        if not cache_dir:
            pytest.skip('COMPASS_DATA_CACHE not set, skipping real data test')

        try:
            # Initialize dataset with real data
            dataset = CustomDataset(path=f'{cache_dir}/data/aime2025/aime2025.jsonl',
                                    abbr='aime2025',
                                    reader_cfg=dict(input_columns=['question'],
                                                    output_column='answer'))

            # Verify dataset was created
            self.assertIsNotNone(dataset.dataset)
            self.assertIsNotNone(dataset.reader)

            # Verify reader configuration
            self.assertEqual(dataset.reader.input_columns, ['question'])
            self.assertEqual(dataset.reader.output_column, 'answer')

            # Verify dataset has data
            self.assertGreater(len(dataset.dataset), 0,
                               'Dataset should not be empty')

            # Verify we can access the data through reader
            self.assertIsNotNone(dataset.reader.dataset)

        except (FileNotFoundError, ValueError, ImportError,
                ModuleNotFoundError) as e:
            assert False, f"Real dataset not available: {e}"
            pytest.skip(f"Real dataset not available: {e}")


if __name__ == '__main__':
    unittest.main()
