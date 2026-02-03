# OpenCompass Dataset Unit Testing Guide

This document provides guidelines and standards for adding unit tests to dataset implementations in `opencompass/datasets/`.

## Directory Structure

Test files should be placed in the `tests/dataset/` directory, following the naming convention `test_<dataset_name>.py`.

```
tests/
  dataset/
    test_<dataset_name>.py
    test_humaneval.py  # Example
    test_beyondaime.py  # Example
```

## Test Coverage

Dataset unit tests should cover the following aspects:

### 1. Dataset Loader Tests

Test the dataset's `load` static method to ensure:
- Data can be loaded correctly
- Data format meets expectations
- Column names and data structure are correct
- Edge cases are handled (empty data, invalid paths, etc.)

**Example test points:**
- Test that `load` method returns the correct `Dataset` or `DatasetDict`
- Test that dataset column names meet expectations
- Test dataset sample count
- Test data transformation logic (e.g., column renaming, data cleaning)

### 2. Dataset Reader Tests

Test the initialization and usage of `DatasetReader`:
- Test `input_columns` and `output_column` configuration
- Test that reader can correctly read the dataset
- Test train/test split handling

### 3. Postprocessor Tests (if applicable)

If the dataset has a `postprocess` function, test:
- Handling of various input formats
- Edge cases (empty strings, special characters, etc.)
- Correctness of output format

Refer to `test_humaneval.py` for postprocessor test examples.

### 4. Evaluator Tests (if custom Evaluator exists)

If the dataset has a custom `Evaluator`, test:
- Correctness of the `score` method
- Prediction and reference value matching logic
- Edge case handling
- **Evaluation result structure validation**: Verify that the result dictionary contains all required fields
- **Metrics calculation**: Verify accuracy, precision, recall, F1, etc. are calculated correctly
- **Result consistency**: Verify that statistics in the result are consistent (e.g., accuracy matches correct/total)
- **Details completeness**: Verify that details array matches the number of samples

For datasets using `CascadeEvaluator` (e.g., AIME2025), test:
- Rule-based evaluator results (e.g., `MATHVerifyEvaluator`)
- LLM evaluator results (e.g., `GenericLLMEvaluator`)
- Cascade mode vs parallel mode behavior
- Final correctness determination logic
- All cascade statistics (rule_accuracy, llm_accuracy, final_accuracy, etc.)

### 5. Integration Tests

Test integration of dataset with reader, inferencer, and evaluator:
- Test complete data flow (load -> read -> infer -> eval)
- Test correctness of configuration files

## Test Writing Standards

### 1. Use unittest Framework

```python
import unittest
from datasets import Dataset
from opencompass.datasets import YourDataset

class TestYourDataset(unittest.TestCase):
    def test_load(self):
        # Test code
        pass
```

### 2. Use Mock for External Dependencies

For datasets that need to load data from HuggingFace or other external sources, use `unittest.mock` to simulate:

```python
from unittest.mock import patch, MagicMock

@patch('datasets.load_dataset')
def test_load_with_mock(self, mock_load_dataset):
    # Create mock data
    mock_dataset = MagicMock()
    mock_dataset.rename_column.return_value = mock_dataset
    mock_load_dataset.return_value = mock_dataset
    
    # Test
    result = YourDataset.load(path='test_path')
    mock_load_dataset.assert_called_once()
```

### 3. Use Temporary Files for File Reading Tests

For datasets that read from local files, use temporary files:

```python
import tempfile
import json

def test_load_from_file(self):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        # Write test data
        json.dump({'test': 'data'}, f)
        f.write('\n')
        f.flush()
        
        # Test loading
        result = YourDataset.load(path=f.name)
        # Assertions
```

### 4. Test Data Validation

Ensure test data meets expected format:

```python
def test_load_returns_dataset(self):
    dataset = YourDataset.load(path='test_path')
    self.assertIsInstance(dataset, Dataset)
    self.assertGreater(len(dataset), 0)
    self.assertIn('expected_column', dataset.column_names)
```

### 5. Test Exception Handling

Test error case handling:

```python
def test_load_invalid_path(self):
    with self.assertRaises(FileNotFoundError):
        YourDataset.load(path='non_existent_path')
```

## Test Examples

### Simple Dataset Tests (e.g., BeyondAIME)

```python
import unittest
from unittest.mock import patch, MagicMock
from datasets import Dataset
from opencompass.datasets import BeyondAIMEDataset

class TestBeyondAIMEDataset(unittest.TestCase):
    
    @patch('datasets.load_dataset')
    def test_load_renames_column(self, mock_load_dataset):
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.rename_column.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        # Execute load
        result = BeyondAIMEDataset.load(path='test_path')
        
        # Verify
        mock_load_dataset.assert_called_once_with(path='test_path', split='test')
        mock_dataset.rename_column.assert_called_once_with('problem', 'question')
        self.assertEqual(result, mock_dataset)
    
    @patch('datasets.load_dataset')
    def test_load_returns_dataset(self, mock_load_dataset):
        from datasets import Dataset
        # Create real Dataset object for testing
        test_data = [{'problem': 'test question', 'answer': 'test answer'}]
        mock_dataset = Dataset.from_list(test_data)
        mock_load_dataset.return_value = mock_dataset
        
        result = BeyondAIMEDataset.load(path='test_path')
        
        self.assertIsInstance(result, Dataset)
        self.assertIn('question', result.column_names)
        self.assertNotIn('problem', result.column_names)
```

### JSONL File Dataset Tests (e.g., AIME2025)

For datasets that load from JSONL files using `CustomDataset`:

```python
import json
import tempfile
import unittest
from unittest.mock import patch
from datasets import Dataset
from opencompass.datasets.custom import CustomDataset

class TestAime2025Dataset(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = [
            {
                'question': 'What is the value of $\\sqrt{16}$?',
                'answer': '4',
                'origin_prompt': 'What is the value of $\\sqrt{16}$?',
                'gold_answer': '4'
            }
        ]
    
    def _create_temp_jsonl_file(self, data):
        """Create a temporary JSONL file with test data."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False, encoding='utf-8'
        )
        for item in data:
            temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        temp_file.close()
        return temp_file.name
    
    @patch('opencompass.utils.get_data_path')
    def test_load_reads_jsonl_file(self, mock_get_data_path):
        """Test that load method reads JSONL file correctly."""
        # Create temporary JSONL file
        temp_file = self._create_temp_jsonl_file(self.test_data)
        mock_get_data_path.return_value = temp_file
        
        try:
            # Execute load
            result = CustomDataset.load(path='test_path')
            
            # Verify dataset was loaded correctly
            self.assertIsInstance(result, Dataset)
            self.assertEqual(len(result), 1)
            self.assertIn('question', result.column_names)
            self.assertIn('answer', result.column_names)
            self.assertEqual(result[0]['question'], self.test_data[0]['question'])
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    @patch('opencompass.utils.get_data_path')
    def test_dataset_initialization(self, mock_get_data_path):
        """Test that dataset can be initialized with proper configuration."""
        temp_file = self._create_temp_jsonl_file(self.test_data)
        mock_get_data_path.return_value = temp_file
        
        try:
            dataset = CustomDataset(
                path='test_path',
                abbr='aime2025_test',
                reader_cfg=dict(input_columns=['question'], output_column='answer')
            )
            
            # Verify dataset was created
            self.assertIsNotNone(dataset.dataset)
            self.assertIsNotNone(dataset.reader)
            self.assertEqual(dataset.reader.input_columns, ['question'])
            self.assertEqual(dataset.reader.output_column, 'answer')
        finally:
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)
```

### Dataset Tests with Postprocessor (e.g., HumanEval)

Refer to `test_humaneval.py` for testing various input format handling.

### Dataset Tests with Custom Evaluator

#### Simple Evaluator Test

```python
import unittest
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.datasets import YourDataset, YourEvaluator

class TestYourEvaluator(unittest.TestCase):
    
    def test_score_correct_prediction(self):
        evaluator = YourEvaluator()
        predictions = ['A']
        references = ['A']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['score'], 1.0)
    
    def test_score_incorrect_prediction(self):
        evaluator = YourEvaluator()
        predictions = ['A']
        references = ['B']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['score'], 0.0)
```

#### Cascade Evaluator Test (e.g., AIME2025)

For datasets using `CascadeEvaluator`, test the evaluation result structure and metrics:

```python
import unittest

class TestAime2025EvalResultValidation(unittest.TestCase):
    """Test cases for AIME2025 evaluation result validation."""
    
    def test_result_structure(self):
        """Test that evaluation result has correct structure."""
        result = {
            'accuracy': 80.0,
            'cascade_stats': {
                'total_samples': 5,
                'rule_correct': 4,
                'rule_accuracy': 80.0,
                'llm_evaluated': 1,
                'llm_correct': 0,
                'llm_accuracy': 0.0,
                'final_correct': 4,
                'final_accuracy': 80.0,
                'parallel_mode': False,
            },
            'details': [{'rule_evaluation': {'correct': True}}] * 5
        }
        
        # Validate structure
        self.assertIn('accuracy', result)
        self.assertIn('cascade_stats', result)
        self.assertIn('details', result)
        
        # Validate cascade_stats
        stats = result['cascade_stats']
        required_keys = [
            'total_samples', 'rule_correct', 'rule_accuracy',
            'llm_evaluated', 'llm_correct', 'llm_accuracy',
            'final_correct', 'final_accuracy', 'parallel_mode'
        ]
        for key in required_keys:
            self.assertIn(key, stats, f"Missing key: {key}")
    
    def test_accuracy_calculation(self):
        """Test that accuracy is calculated correctly."""
        result = {
            'accuracy': 80.0,
            'cascade_stats': {
                'total_samples': 5,
                'final_correct': 4,
                'final_accuracy': 80.0,
            }
        }
        
        expected_accuracy = (4 / 5) * 100
        self.assertAlmostEqual(result['accuracy'], expected_accuracy, places=1)
        self.assertAlmostEqual(
            result['cascade_stats']['final_accuracy'],
            expected_accuracy,
            places=1
        )
    
    def test_cascade_mode_result(self):
        """Test cascade mode evaluation result."""
        result = {
            'accuracy': 100.0,
            'cascade_stats': {
                'total_samples': 3,
                'rule_correct': 2,
                'rule_accuracy': 66.67,
                'llm_evaluated': 1,  # Only failed samples
                'llm_correct': 1,
                'llm_accuracy': 100.0,
                'final_correct': 3,
                'final_accuracy': 100.0,
                'parallel_mode': False,
            },
            'details': [
                {
                    'rule_evaluation': {'correct': True},
                    'llm_evaluation': None,
                    'cascade_correct': True
                },
                {
                    'rule_evaluation': {'correct': True},
                    'llm_evaluation': None,
                    'cascade_correct': True
                },
                {
                    'rule_evaluation': {'correct': False},
                    'llm_evaluation': {'llm_correct': True},
                    'cascade_correct': True
                }
            ]
        }
        
        # Validate cascade logic
        self.assertEqual(result['cascade_stats']['parallel_mode'], False)
        self.assertEqual(result['cascade_stats']['llm_evaluated'], 1)
        
        # Check cascade_correct logic
        for detail in result['details']:
            rule_correct = detail['rule_evaluation'].get('correct', False)
            llm_correct = detail.get('llm_evaluation', {}).get('llm_correct', False) if detail.get('llm_evaluation') else False
            cascade_correct = detail['cascade_correct']
            self.assertEqual(cascade_correct, rule_correct or llm_correct)
    
    def test_result_statistics_consistency(self):
        """Test that statistics in result are consistent."""
        result = {
            'accuracy': 60.0,
            'cascade_stats': {
                'total_samples': 5,
                'rule_correct': 3,
                'rule_accuracy': 60.0,
                'llm_evaluated': 2,
                'llm_correct': 0,
                'llm_accuracy': 0.0,
                'final_correct': 3,
                'final_accuracy': 60.0,
                'parallel_mode': False,
            },
            'details': [{'rule_evaluation': {'correct': i < 3}} for i in range(5)]
        }
        
        stats = result['cascade_stats']
        
        # Check accuracy calculations
        self.assertAlmostEqual(
            stats['rule_accuracy'],
            (stats['rule_correct'] / stats['total_samples']) * 100,
            places=1
        )
        
        if stats['llm_evaluated'] > 0:
            self.assertAlmostEqual(
                stats['llm_accuracy'],
                (stats['llm_correct'] / stats['llm_evaluated']) * 100,
                places=1
            )
        
        # Check that final_accuracy matches top-level accuracy
        self.assertAlmostEqual(
            result['accuracy'],
            stats['final_accuracy'],
            places=1
        )
    
    def test_details_count_matches_total_samples(self):
        """Test that details count matches total_samples."""
        total_samples = 10
        result = {
            'cascade_stats': {'total_samples': total_samples},
            'details': [{}] * total_samples
        }
        
        self.assertEqual(len(result['details']), result['cascade_stats']['total_samples'])
    
    def test_result_metrics_completeness(self):
        """Test that all required metrics are present in result."""
        required_metrics = [
            'accuracy',
            'cascade_stats.total_samples',
            'cascade_stats.rule_correct',
            'cascade_stats.rule_accuracy',
            'cascade_stats.llm_evaluated',
            'cascade_stats.llm_correct',
            'cascade_stats.llm_accuracy',
            'cascade_stats.final_correct',
            'cascade_stats.final_accuracy',
            'cascade_stats.parallel_mode',
            'details',
        ]
        
        result = {
            'accuracy': 80.0,
            'cascade_stats': {
                'total_samples': 5,
                'rule_correct': 4,
                'rule_accuracy': 80.0,
                'llm_evaluated': 1,
                'llm_correct': 0,
                'llm_accuracy': 0.0,
                'final_correct': 4,
                'final_accuracy': 80.0,
                'parallel_mode': False,
            },
            'details': [{}] * 5
        }
        
        # Check top-level accuracy
        self.assertIn('accuracy', result)
        self.assertIsInstance(result['accuracy'], (int, float))
        
        # Check cascade_stats
        self.assertIn('cascade_stats', result)
        stats = result['cascade_stats']
        for key in ['total_samples', 'rule_correct', 'rule_accuracy',
                   'llm_evaluated', 'llm_correct', 'llm_accuracy',
                   'final_correct', 'final_accuracy', 'parallel_mode']:
            self.assertIn(key, stats, f"Missing metric: {key}")
        
        # Check details
        self.assertIn('details', result)
        self.assertIsInstance(result['details'], list)
```

## Running Tests

### Prerequisites

Before running tests, ensure all necessary dependencies are installed:

```bash
pip install -r requirements.txt
```

Some tests may require additional dependencies such as `evaluate`, `transformers`, etc.

### Run a Single Test File

```bash
python -m pytest tests/dataset/test_<dataset_name>.py -v
```

### Run All Dataset Tests

```bash
python -m pytest tests/dataset/ -v
```

### Run a Specific Test Method

```bash
python -m pytest tests/dataset/test_<dataset_name>.py::TestYourDataset::test_load -v
```

### Testing with Real Dataset Files

You can test with real dataset files by setting the `COMPASS_DATA_CACHE` environment variable:

```bash
export COMPASS_DATA_CACHE=/path/to/data/cache
python -m pytest tests/dataset/test_aime2025.py::TestAime2025Dataset::test_load_with_real_data -v
```

The test will automatically skip if `COMPASS_DATA_CACHE` is not set or if the dataset file is not found.

Example test using real data:

```python
def test_load_with_real_data(self):
    """Test loading dataset with real data from COMPASS_DATA_CACHE."""
    import pytest
    
    # Check if COMPASS_DATA_CACHE is set
    cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
    if not cache_dir:
        pytest.skip("COMPASS_DATA_CACHE not set, skipping real data test")
    
    try:
        # Load real dataset
        dataset = CustomDataset.load(path='opencompass/aime2025')
        
        # Verify dataset was loaded
        self.assertIsInstance(dataset, Dataset)
        self.assertGreater(len(dataset), 0)
        self.assertIn('question', dataset.column_names)
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"Real dataset not found: {e}")
```

### Handling Import Errors

If you encounter import errors (e.g., `ModuleNotFoundError: No module named 'evaluate'`), you can:

1. Install missing dependencies
2. Use mocks to avoid importing the complete opencompass package
3. Use `pytest.skip()` in tests to skip tests that require specific dependencies

## Best Practices

1. **Keep tests independent**: Each test should run independently without depending on the state of other tests
2. **Use descriptive test names**: Test method names should clearly describe what is being tested
3. **Test edge cases**: Include empty data, invalid inputs, exception cases, etc.
4. **Use fixtures**: For repeated test data, use fixtures or setUp methods
5. **Mock external dependencies**: Avoid actually downloading or accessing external resources in tests
6. **Keep tests fast**: Unit tests should execute quickly, avoiding long-running operations
7. **Test coverage**: Try to cover all code paths, including error handling

## Notes

1. **Data paths**: Use mocks or temporary files in tests, avoid depending on actual data file paths
2. **Environment variables**: Note that environment variables like `DATASET_SOURCE` may affect dataset loading logic
3. **Dependencies**: Ensure all necessary dependencies are installed in the test environment
4. **Version compatibility**: Tests should pass under different versions of dependency libraries

## Reference Examples

- `tests/dataset/test_humaneval.py` - Postprocessor test example
- `tests/dataset/test_local_datasets.py` - Dataset loading test example
- `tests/dataset/test_beyondaime.py` - Simple dataset test example (HuggingFace dataset)
- `tests/dataset/test_aime2025.py` - JSONL file dataset test example (CustomDataset)
- `tests/dataset/test_aime2025_eval.py` - Evaluator result validation test example (CascadeEvaluator)