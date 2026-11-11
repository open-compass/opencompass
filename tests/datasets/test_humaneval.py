import unittest

from opencompass.datasets.humaneval import (
    _call_evaluate_functional_correctness, _supports_ignore_incomplete,
    humaneval_chat_postprocess, humaneval_postprocess_v2,
    humaneval_postprocess_v3, logger as humaneval_logger)


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

    def test_chat_extracts_unclosed_code_block(self):
        raw = '\n'.join([
            'Here is the implementation:',
            '```python',
            'def truncate_number(number: float) -> float:',
            '    return number - int(number)',
        ])

        self.assertEqual(
            humaneval_chat_postprocess(raw),
            'def truncate_number(number: float) -> float:\n'
            '    return number - int(number)\n')

    def test_chat_extracts_code_after_reasoning(self):
        raw = '\n'.join([
            'We need to subtract the integer part.',
            'The final answer is:',
            'def truncate_number(number: float) -> float:',
            '    return number - int(number)',
        ])

        self.assertEqual(
            humaneval_chat_postprocess(raw),
            'def truncate_number(number: float) -> float:\n'
            '    return number - int(number)\n')

    def test_chat_trims_trailing_explanation(self):
        raw = '\n'.join([
            'Here is the implementation:',
            'def truncate_number(number: float) -> float:',
            '    return number - int(number)',
            '',
            'This solves the task.',
        ])

        self.assertEqual(
            humaneval_chat_postprocess(raw),
            'def truncate_number(number: float) -> float:\n'
            '    return number - int(number)\n')

    def test_chat_keeps_plain_function_body(self):
        raw = '    return x - int(x)\n'

        self.assertEqual(humaneval_chat_postprocess(raw),
                         'return x - int(x)\n')


class TestHumanEvalEvaluatorCompat(unittest.TestCase):

    def test_supports_ignore_incomplete(self):

        def supported(value, ignore_incomplete=False):
            return value

        def unsupported(value):
            return value

        def variadic(value, **kwargs):
            return value

        self.assertTrue(_supports_ignore_incomplete(supported))
        self.assertFalse(_supports_ignore_incomplete(unsupported))
        self.assertTrue(_supports_ignore_incomplete(variadic))

    def test_unsupported_function_does_not_receive_flag(self):

        def evaluate(value, n_workers=4):
            return {'value': value, 'n_workers': n_workers}

        result = _call_evaluate_functional_correctness(
            evaluate, 'sample', n_workers=2, ignore_incomplete=True)

        self.assertEqual(result, {'value': 'sample', 'n_workers': 2})

    def test_supported_function_receives_flag(self):

        def evaluate(value, ignore_incomplete=False):
            return ignore_incomplete

        result = _call_evaluate_functional_correctness(
            evaluate, 'sample', ignore_incomplete=True)

        self.assertTrue(result)

    def test_variadic_function_receives_flag(self):

        def evaluate(value, **kwargs):
            return kwargs

        result = _call_evaluate_functional_correctness(
            evaluate, 'sample', ignore_incomplete=True)

        self.assertTrue(result['ignore_incomplete'])

    def test_unsupported_function_warns_when_flag_requested(self):

        def evaluate(value):
            return value

        with self.assertLogs(humaneval_logger, level='WARNING'):
            _call_evaluate_functional_correctness(
                evaluate, 'sample', ignore_incomplete=True)


if __name__ == '__main__':
    unittest.main()
