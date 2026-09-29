"""Unit tests for GSM8K dataset.

Covers text postprocessors and the Gsm8kEvaluator. The module is loaded via
importlib to avoid importing the full opencompass package and its heavy
dependencies (torch, transformers, etc.).
"""

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_gsm8k_module():
    module_name = 'opencompass.datasets.gsm8k'
    if module_name in sys.modules:
        return sys.modules[module_name]

    root_dir = Path(__file__).resolve().parents[2]
    module_path = root_dir / 'opencompass' / 'datasets' / 'gsm8k.py'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


gsm8k_module = _load_gsm8k_module()
gsm8k_postprocess = gsm8k_module.gsm8k_postprocess
gsm8k_dataset_postprocess = gsm8k_module.gsm8k_dataset_postprocess
Gsm8kEvaluator = gsm8k_module.Gsm8kEvaluator


class TestGsm8kPostprocess(unittest.TestCase):
    """Test cases for gsm8k_postprocess."""

    def test_extracts_last_number(self):
        self.assertEqual(gsm8k_postprocess('The answer is 42'), '42')

    def test_prefers_last_number(self):
        self.assertEqual(gsm8k_postprocess('3.14 and 42'), '42')

    def test_handles_negative_number(self):
        self.assertEqual(gsm8k_postprocess('result is -5'), '-5')

    def test_returns_null_when_no_number(self):
        self.assertEqual(gsm8k_postprocess('no numbers here'), 'NULL')

    def test_extracts_integer_from_float_text(self):
        self.assertEqual(gsm8k_postprocess('value 3.14'), '3.14')

    def test_empty_string_returns_null(self):
        self.assertEqual(gsm8k_postprocess(''), 'NULL')


class TestGsm8kDatasetPostprocess(unittest.TestCase):
    """Test cases for gsm8k_dataset_postprocess."""

    def test_extracts_after_answer_separator(self):
        self.assertEqual(gsm8k_dataset_postprocess('final #### 1234'), '1234')

    def test_removes_thousands_separator(self):
        self.assertEqual(gsm8k_dataset_postprocess('#### 1,234'), '1234')

    def test_keeps_decimal(self):
        self.assertEqual(gsm8k_dataset_postprocess('#### 12.5'), '12.5')


class TestGsm8kEvaluator(unittest.TestCase):
    """Test cases for Gsm8kEvaluator."""

    def setUp(self):
        self.evaluator = Gsm8kEvaluator()

    def test_exact_match(self):
        result = self.evaluator.score(['42'], ['42'])
        self.assertEqual(result['accuracy'], 100.0)
        self.assertTrue(result['details'][0]['correct'])

    def test_float_int_tolerance(self):
        result = self.evaluator.score(['42.0'], ['42'])
        self.assertEqual(result['accuracy'], 100.0)
        self.assertTrue(result['details'][0]['correct'])

    def test_float_equal_within_epsilon(self):
        result = self.evaluator.score(['42.000001'], ['42'])
        self.assertEqual(result['accuracy'], 100.0)

    def test_wrong_answer(self):
        result = self.evaluator.score(['100'], ['42'])
        self.assertEqual(result['accuracy'], 0.0)
        self.assertFalse(result['details'][0]['correct'])

    def test_partial_accuracy(self):
        result = self.evaluator.score(['42', '100'], ['42', '42'])
        self.assertEqual(result['accuracy'], 50.0)

    def test_mismatched_length_returns_error(self):
        result = self.evaluator.score(['1', '2'], ['1'])
        self.assertIn('error', result)

    def test_non_numeric_pred_is_false(self):
        self.assertFalse(self.evaluator.is_equal('abc', '42'))

    def test_none_pred_is_false(self):
        self.assertFalse(self.evaluator.is_equal(None, '42'))


if __name__ == '__main__':
    unittest.main()
