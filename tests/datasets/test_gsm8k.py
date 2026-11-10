"""Unit tests for GSM8K dataset.

Covers text postprocessors and the Gsm8kEvaluator.
"""

import unittest

from opencompass.datasets.gsm8k import (Gsm8kEvaluator,
                                        gsm8k_dataset_postprocess,
                                        gsm8k_postprocess)


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

    def test_strips_thousands_separators(self):
        cases = {
            r'\boxed{$9{,}500}': '9500',
            r'\boxed{8,000}': '8000',
            'The answer is $9,500.': '9500',
            'The answer is 1,450,000': '1450000',
            'The total is $1,234,567.89': '1234567.89',
            r'Thus, Jill earns \(\boxed{\$57,500}\).': '57500',
            r'Josh incurred a loss: $$ \boxed{-10,000} $$': '-10000',
            r'The distance is \[ \boxed{180,000 \text{ meters}} \]': '180000',
            ('$22,500+$35,000 = $57,500 per year. '
             'The answer is $57,500'): '57500',
            'The answer is 2,000.\n\nQuestion: He has 5,000 coins.': '2000',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(gsm8k_postprocess(text), expected)

    def test_keeps_non_thousands_commas(self):
        # A comma is only a thousands separator when exactly three digits
        # follow, so lists, coordinates and decimal commas must not be merged.
        cases = {
            'The side lengths are 3,4,5': '5',
            'The point is (3,4)': '4',
            'Days 1,2 she ran 5 miles; total 12,15': '15',
            'So x = 2,5': '5',
            '12,34': '34',
            '1,2345': '2345',
            'Sides 3,4,5 give area 6, so it costs $1,200': '1200',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(gsm8k_postprocess(text), expected)


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
