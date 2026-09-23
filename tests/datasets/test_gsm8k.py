"""Unit tests for GSM8K postprocessors (incl. #2647 thousands separators)."""

import unittest

from opencompass.datasets.gsm8k import (gsm8k_dataset_postprocess,
                                        gsm8k_postprocess)


class TestGsm8kPostprocess(unittest.TestCase):

    def test_extracts_last_number(self):
        self.assertEqual(gsm8k_postprocess('The answer is 42'), '42')

    def test_prefers_last_number(self):
        self.assertEqual(gsm8k_postprocess('3.14 and 42'), '42')

    def test_handles_negative_number(self):
        self.assertEqual(gsm8k_postprocess('result is -5'), '-5')

    def test_returns_null_when_no_number(self):
        self.assertEqual(gsm8k_postprocess('no numbers here'), 'NULL')

    def test_plain_boxed_integer(self):
        self.assertEqual(gsm8k_postprocess(r'\boxed{9500}'), '9500')

    def test_ascii_thousands_separator_in_boxed(self):
        # open-compass/opencompass#2647
        self.assertEqual(gsm8k_postprocess(r'\boxed{8,000}'), '8000')
        self.assertEqual(gsm8k_postprocess(r'\boxed{10,000}'), '10000')

    def test_latex_thousands_separator_in_boxed(self):
        self.assertEqual(gsm8k_postprocess(r'\boxed{$9{,}500}'), '9500')

    def test_latex_thousands_separator_in_prose(self):
        self.assertEqual(gsm8k_postprocess(r'The answer is $9{,}500.'), '9500')

    def test_boxed_preferred_over_earlier_numbers(self):
        text = r'Step 1 uses 3 apples. Final: \boxed{8,000}'
        self.assertEqual(gsm8k_postprocess(text), '8000')

    def test_digitless_boxed_does_not_hide_later_number(self):
        # Review on #2648: a box with no digit must not discard the rest.
        self.assertEqual(
            gsm8k_postprocess(r'\boxed{x} and the total is 42'), '42')
        self.assertEqual(gsm8k_postprocess(r'\boxed{} the total is 42'), '42')

    def test_repeated_latex_comma_separator(self):
        self.assertEqual(gsm8k_postprocess(r'\boxed{1{,}234{,}567}'),
                         '1234567')

    def test_latex_thin_space_and_tilde_separators(self):
        self.assertEqual(gsm8k_postprocess(r'\boxed{9\,500}'), '9500')
        self.assertEqual(gsm8k_postprocess(r'\boxed{9\;500}'), '9500')
        self.assertEqual(gsm8k_postprocess(r'\boxed{9~500}'), '9500')
        self.assertEqual(gsm8k_postprocess(r'\boxed{1\,234\,567}'), '1234567')

    def test_space_between_digits_is_not_a_separator(self):
        # "9 500" can be a list; only the last group is kept.
        self.assertEqual(gsm8k_postprocess(r'\boxed{9 500}'), '500')


class TestGsm8kDatasetPostprocess(unittest.TestCase):

    def test_extracts_after_answer_separator(self):
        self.assertEqual(gsm8k_dataset_postprocess('final #### 1234'), '1234')

    def test_removes_thousands_separator(self):
        self.assertEqual(gsm8k_dataset_postprocess('#### 1,234'), '1234')


if __name__ == '__main__':
    unittest.main()
