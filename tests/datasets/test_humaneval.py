import unittest

from opencompass.datasets.humaneval import (humaneval_postprocess_v2,
                                            humaneval_postprocess_v3)


class TestHumanevalPostprocess(unittest.TestCase):

    def test_v2_returns_plain_text_without_code_fence(self):
        raw = '    return x - int(x)\n'

        self.assertEqual(humaneval_postprocess_v2(raw),
                         'return x - int(x)\n')

    def test_v2_extracts_python_code_block(self):
        raw = '\n'.join([
            'Here is the implementation:',
            '```python',
            '    return x - int(x)',
            '```',
            'This solves the task.',
        ])

        self.assertEqual(humaneval_postprocess_v2(raw),
                         'return x - int(x)\n')

    def test_v2_extracts_bare_code_block(self):
        raw = '\n'.join([
            '```',
            '    return x - int(x)',
            '```',
        ])

        self.assertEqual(humaneval_postprocess_v2(raw),
                         'return x - int(x)\n')

    def test_v2_uses_first_code_block(self):
        raw = '\n'.join([
            '```python',
            '    return "first"',
            '```',
            '```python',
            '    return "second"',
            '```',
        ])

        self.assertEqual(humaneval_postprocess_v2(raw),
                         'return "first"\n')

    def test_v3_uses_last_code_block(self):
        raw = '\n'.join([
            '```python',
            '    return "first"',
            '```',
            '```python',
            '    return "second"',
            '```',
        ])

        self.assertEqual(humaneval_postprocess_v3(raw),
                         'return "second"\n')

    def test_v3_matches_v2_for_single_code_block(self):
        raw = '\n'.join([
            '```python',
            '    value = x - int(x)',
            '    return value',
            '```',
        ])

        expected = 'value = x - int(x)\n    return value\n'
        self.assertEqual(humaneval_postprocess_v2(raw), expected)
        self.assertEqual(humaneval_postprocess_v3(raw), expected)


if __name__ == '__main__':
    unittest.main()
