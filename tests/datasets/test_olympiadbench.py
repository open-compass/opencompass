import unittest

from opencompass.datasets.OlympiadBench import MathJudger


class TestMathJudger(unittest.TestCase):

    def setUp(self):
        self.judger = MathJudger()

    def test_factored_product_equals_expanded_form(self):
        self.assertTrue(self.judger.judge('(x-1)(x+1)', '(x^2-1)'))
        self.assertFalse(self.judger.judge('(x-1)(x+1)', '(x^2+1)'))

    def test_intervals_are_compared_by_bounds(self):
        self.assertTrue(self.judger.judge('(1,2)', '(1, 2)'))
        self.assertFalse(self.judger.judge('(1,2)', '(1,3)'))
        self.assertFalse(self.judger.judge('[0,1]', '(0,1)'))


if __name__ == '__main__':
    unittest.main()
