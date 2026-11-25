import unittest

from opencompass.datasets.math import MATHEvaluator

# (version, prediction, gold, expected)
EQUIVALENCE_CASES = [
    # answers that differ must not be scored as equal
    ('v1', 'x<3', 'x \\le 3', False),
    ('v2', 'x<3', 'x \\le 3', False),
    ('v1', '12', '1,2', False),
    ('v2', '23', '2\\cdot 3', False),
    ('v2', '5, 9', '5 \\cdot 9', False),
    ('v2', '34', '3 and 4', False),
    ('v2', '3\\mathbf{and}4', '34', False),
    ('v1', '3{,}4', '34', False),
    ('v1', '3{,}4', '3,4', True),
    ('v1', '(123, 456)', '(123456)', False),
    ('v1', '1,234,5678', '12345678', False),
    # same answer written differently must still match
    ('v1', 'x \\le 3', 'x<=3', True),
    ('v1', 'x \\leq 3', 'x<=3', True),
    ('v1', 'x \\le 3', 'x \\leq 3', True),
    ('v1', '100,000', '100000', True),
    ('v1', '1{,}000', '1000', True),
    ('v1', '(1{,}000)', '(1000)', True),
    ('v1', '-1,000', '-1000', True),
    ('v1', '1,000.5', '1000.5', True),
    ('v1', '0.5', '\\frac{1}{2}', True),
    ('v2', '2\\cdot 3', '2*3', True),
    ('v2', '3 and 4', '3,4', True),
    ('v2', '\\frac{1}{2}', '\\dfrac{1}{2}', True),
]


class TestMATHEvaluatorEquivalence(unittest.TestCase):

    def test_is_equiv(self):
        evaluators = {
            'v1': MATHEvaluator(version='v1'),
            'v2': MATHEvaluator(version='v2'),
        }
        for version, pred, gold, expected in EQUIVALENCE_CASES:
            with self.subTest(version=version, pred=pred, gold=gold):
                self.assertIs(expected,
                              evaluators[version].is_equiv(pred, gold))


if __name__ == '__main__':
    unittest.main()
