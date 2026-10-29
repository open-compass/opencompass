import unittest

from opencompass.datasets.mbpp import MBPPEvaluator


class TestMBPPEvaluator(unittest.TestCase):

    def test_process_answer_uses_first_generated_program(self):
        raw = """ 'def target(x):
    return x + 1'
[DONE]

[BEGIN]
 'def wrong(x):
    return x - 1'
[DONE]
"""

        self.assertEqual(MBPPEvaluator()._process_answer(raw),
                         'def target(x):\n    return x + 1')


if __name__ == '__main__':
    unittest.main()
