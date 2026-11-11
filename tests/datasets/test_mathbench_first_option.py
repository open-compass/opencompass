"""Unit tests for mathbench.first_option_postprocess."""

import unittest

from opencompass.datasets.mathbench import first_option_postprocess


class TestMathbenchFirstOptionPostprocess(unittest.TestCase):

    def test_only_option_is_correct_pattern(self):
        # The definitive "只有选项X是对" must win over earlier distractors.
        # cushion=False ensures the result comes from this pattern rather than
        # the loose fallback patterns.
        cases = {
            '选项A错误，选项B也不对，只有选项C是对': 'C',
            '选项D是干扰项，只有选项A是对': 'A',
            '先看B，B不正确，只有选项D是对': 'D',
        }
        for text, expected in cases.items():
            for cushion in (True, False):
                with self.subTest(text=text, cushion=cushion):
                    answer, _ = first_option_postprocess(text,
                                                         'ABCD',
                                                         cushion=cushion)
                    self.assertEqual(answer, expected)

    def test_guxuan_pattern(self):
        cases = {
            '故选 B': 'B',
            '故C': 'C',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                answer, _ = first_option_postprocess(text,
                                                     'ABCD',
                                                     cushion=False)
                self.assertEqual(answer, expected)


if __name__ == '__main__':
    unittest.main()
