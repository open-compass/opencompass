import unittest

from opencompass.datasets.humaneval import humaneval_postprocess


def run_humaneval_check(completion):
    program = [
        'def get_fraction(x: float) -> float:',
        humaneval_postprocess(completion),
        '',
        'assert get_fraction(1.28) == 0.28',
        'assert get_fraction(1.0) == 0.0',
    ]
    program = '\n'.join(program)
    exec(program)


class TestHumaneval(unittest.TestCase):

    def test_vanilla(self):
        raw = '    return x - int(x)'
        run_humaneval_check(raw)

    def test_python_quote(self):
        lines = [
            '```python',
            '    return x - int(x)',
            '```',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_bare_quote(self):
        lines = [
            '```',
            '    return x - int(x)',
            '```',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_error_space_quote(self):
        lines = [
            '```',
            '  return x - int(x)',
            '```',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_import_1(self):
        lines = [
            'import numpy as np',
            'import math',
            'from typing import List',
            '',
            'def func(x):',
            '    return x - int(x)',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_import_2(self):
        lines = [
            'from typing import List',
            'import numpy as np',
            'import math',
            'def func(x):',
            '    return x - int(x)',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_import_3(self):
        lines = [
            'import math',
            '',
            '',
            'def func(x):',
            '    return x - int(x)',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_comment(self):
        lines = [
            'def func(x: float) -> float:',
            "    '''",
            '    blah blah blah',
            '    blah blah blah',
            "    '''",
            '    return x - int(x)',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_additional(self):
        lines = [
            '    return x - int(x)',
            '',
            '',
            'def func(x: float) -> float:',
            "    '''",
            '    blah blah blah',
            '    blah blah blah',
            "    '''",
            '    return x - int(x)',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)
