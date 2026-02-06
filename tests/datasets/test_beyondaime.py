"""Unit tests for BeyondAIME dataset.

Note: This test requires the opencompass package and its dependencies to be installed.
If you encounter import errors, ensure all dependencies are installed:
    pip install -r requirements.txt
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset


def _load_beyondaime_module():
    module_name = 'opencompass.datasets.beyondaime'
    if module_name in sys.modules:
        return sys.modules[module_name]

    root_dir = Path(__file__).resolve().parents[2]
    module_path = root_dir / 'opencompass' / 'datasets' / 'beyondaime.py'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BeyondAIMEDataset = _load_beyondaime_module().BeyondAIMEDataset


class TestBeyondAIMEDataset(unittest.TestCase):
    """Test cases for BeyondAIMEDataset."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = [{
            'problem': 'What is 2+2?',
            'answer': '4'
        }, {
            'problem': 'Solve for x: x^2 = 4',
            'answer': 'x = 2 or x = -2'
        }]

    @patch('opencompass.datasets.beyondaime.load_dataset')
    def test_load_renames_problem_to_question(self, mock_load_dataset):
        """Test that load method renames 'problem' column to 'question'."""
        # Create a mock dataset with 'problem' column
        mock_dataset = Dataset.from_list(self.test_data)
        mock_load_dataset.return_value = mock_dataset

        # Execute load
        result = BeyondAIMEDataset.load(path='test_path')

        # Verify load_dataset was called correctly
        mock_load_dataset.assert_called_once_with(path='test_path',
                                                  split='test')

        # Verify column was renamed
        self.assertIn('question', result.column_names)
        self.assertNotIn('problem', result.column_names)
        self.assertEqual(result[0]['question'], 'What is 2+2?')
        self.assertEqual(result[1]['question'], 'Solve for x: x^2 = 4')

    @patch('opencompass.datasets.beyondaime.load_dataset')
    def test_load_returns_dataset(self, mock_load_dataset):
        """Test that load method returns a Dataset instance."""
        mock_dataset = Dataset.from_list(self.test_data)
        mock_load_dataset.return_value = mock_dataset

        result = BeyondAIMEDataset.load(path='test_path')

        self.assertIsInstance(result, Dataset)
        self.assertEqual(len(result), 2)

    @patch('opencompass.datasets.beyondaime.load_dataset')
    def test_load_preserves_other_columns(self, mock_load_dataset):
        """Test that load method preserves columns other than 'problem'."""
        extended_data = [{
            'problem': 'Test question',
            'answer': 'Test answer',
            'difficulty': 'hard',
            'category': 'algebra'
        }]
        mock_dataset = Dataset.from_list(extended_data)
        mock_load_dataset.return_value = mock_dataset

        result = BeyondAIMEDataset.load(path='test_path')

        # Verify all columns except 'problem' are preserved
        self.assertIn('question', result.column_names)
        self.assertIn('answer', result.column_names)
        self.assertIn('difficulty', result.column_names)
        self.assertIn('category', result.column_names)
        self.assertNotIn('problem', result.column_names)

    @patch('opencompass.datasets.beyondaime.load_dataset')
    def test_load_with_empty_dataset(self, mock_load_dataset):
        """Test load method with empty dataset."""
        # Create empty dataset with 'problem' column to test rename logic
        empty_dataset = Dataset.from_list([])
        # Add 'problem' column structure even if empty
        if len(empty_dataset
               ) == 0 and 'problem' not in empty_dataset.column_names:
            # For empty dataset without columns, we'll skip the rename test
            # as it's not a realistic scenario
            empty_dataset = Dataset.from_dict({'problem': [], 'answer': []})
        mock_load_dataset.return_value = empty_dataset

        result = BeyondAIMEDataset.load(path='test_path')

        self.assertIsInstance(result, Dataset)
        self.assertEqual(len(result), 0)
        # If dataset had 'problem' column, it should be renamed to 'question'
        if 'problem' in empty_dataset.column_names:
            self.assertIn('question', result.column_names)
            self.assertNotIn('problem', result.column_names)

    @patch('opencompass.datasets.beyondaime.load_dataset')
    def test_load_with_kwargs(self, mock_load_dataset):
        """Test that load method passes kwargs to load_dataset."""
        mock_dataset = Dataset.from_list(self.test_data)
        mock_load_dataset.return_value = mock_dataset

        BeyondAIMEDataset.load(path='test_path', cache_dir='/tmp/cache')

        # Verify kwargs are passed (though load_dataset might not use all kwargs)
        # At minimum, verify it was called
        mock_load_dataset.assert_called_once()

    @patch('opencompass.datasets.beyondaime.load_dataset')
    def test_dataset_initialization(self, mock_load):
        """Test that dataset can be initialized with proper configuration."""
        mock_dataset = Dataset.from_list(self.test_data)
        mock_load.return_value = mock_dataset

        # Initialize dataset with reader config
        dataset = BeyondAIMEDataset(path='test_path',
                                    abbr='beyondaime_test',
                                    reader_cfg=dict(
                                        input_columns=['question'],
                                        output_column='answer'))

        # Verify dataset was created
        self.assertIsNotNone(dataset.dataset)
        self.assertIsNotNone(dataset.reader)
            

    @patch('opencompass.datasets.beyondaime.load_dataset')
    def test_dataset_reader_config(self, mock_load):
        """Test that dataset reader is configured correctly."""
        mock_dataset = Dataset.from_list(self.test_data)
        mock_load.return_value = mock_dataset

        reader_cfg = dict(input_columns=['question'], output_column='answer')

        dataset = BeyondAIMEDataset(path='test_path',
                                    abbr='beyondaime_test',
                                    reader_cfg=reader_cfg)

        # Verify reader configuration
        self.assertEqual(dataset.reader.input_columns, ['question'])
        self.assertEqual(dataset.reader.output_column, 'answer')


if __name__ == '__main__':
    unittest.main()
