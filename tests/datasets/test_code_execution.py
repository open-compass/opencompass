import unittest

from opencompass.utils.code_execution import (TYPE_AWARE_CLOSE_NAME,
                                              TYPE_AWARE_EQUAL_NAME,
                                              make_assertions_type_aware,
                                              make_humaneval_test_type_aware,
                                              type_aware_close,
                                              type_aware_equal)


class _AlwaysEqual:

    def __eq__(self, other):
        return True


class _AlwaysEqualInt(int):

    def __new__(cls):
        return int.__new__(cls, 999)

    def __eq__(self, other):
        return True


class _AlwaysEqualArith:

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __sub__(self, other):
        return 0

    def __rsub__(self, other):
        return 0


class TestCodeExecution(unittest.TestCase):

    def test_type_aware_equal_rejects_eq_override(self):
        self.assertFalse(type_aware_equal(_AlwaysEqual(), 1))

    def test_type_aware_equal_rejects_nested_eq_override(self):
        self.assertFalse(type_aware_equal([_AlwaysEqualInt()], [1]))

    def test_type_aware_equal_accepts_nested_builtin_value(self):
        self.assertTrue(type_aware_equal({'value': [1, 2]}, {'value': [1, 2]}))

    def test_type_aware_assertion_rejects_eq_override(self):
        test_program = make_assertions_type_aware('assert solve() == 1')
        namespace = {
            'solve': _AlwaysEqual,
            TYPE_AWARE_EQUAL_NAME: type_aware_equal,
        }

        with self.assertRaises(AssertionError):
            exec(test_program, namespace)

    def test_type_aware_assertion_accepts_matching_builtin_value(self):
        test_program = make_assertions_type_aware('assert solve() == 1')
        namespace = {
            'solve': lambda: 1,
            TYPE_AWARE_EQUAL_NAME: type_aware_equal,
        }

        exec(test_program, namespace)

    def test_type_aware_close_rejects_forged_sub(self):
        self.assertFalse(type_aware_close(_AlwaysEqualArith(), 0.33, 1e-6))

    def test_type_aware_close_rejects_float_subclass(self):

        class _Float(float):

            def __new__(cls):
                return float.__new__(cls, 0.33)

            def __sub__(self, other):
                return 0

        self.assertFalse(type_aware_close(_Float(), 0.33, 1e-6))

    def test_type_aware_close_accepts_float_within_tolerance(self):
        self.assertTrue(type_aware_close(0.3300001, 0.33, 1e-6))
        self.assertFalse(type_aware_close(0.5, 0.33, 1e-6))

    def _exec_humaneval(self, completion, test, entry='solve'):
        # Same shape as human_eval.execution.check_correctness: one exec,
        # empty globals, then check(entry_point).
        program = (completion + '\n' + make_humaneval_test_type_aware(test) +
                   '\ncheck(%s)\n' % entry)
        exec(program, {})

    def test_humaneval_forged_equality_is_rejected(self):
        test = 'def check(candidate):\n    assert candidate() == 1\n'
        completion = ('def solve():\n'
                      '    class _Always:\n'
                      '        def __eq__(self, other):\n'
                      '            return True\n'
                      '        def __ne__(self, other):\n'
                      '            return False\n'
                      '    return _Always()\n')

        with self.assertRaises(AssertionError):
            self._exec_humaneval(completion, test)

    def test_humaneval_forged_tolerance_is_rejected(self):
        test = ('def check(candidate):\n'
                '    assert abs(candidate(1.33) - 0.33) < 1e-6\n')
        completion = ('def solve(number):\n'
                      '    class _Always:\n'
                      '        def __eq__(self, other):\n'
                      '            return True\n'
                      '        def __ne__(self, other):\n'
                      '            return False\n'
                      '        def __sub__(self, other):\n'
                      '            return 0\n'
                      '        def __rsub__(self, other):\n'
                      '            return 0\n'
                      '    return _Always()\n')

        with self.assertRaises(AssertionError):
            self._exec_humaneval(completion, test)

    def test_humaneval_honest_wrong_answer_is_rejected(self):
        test = ('def check(candidate):\n'
                '    assert candidate(3.5) == 0.5\n'
                '    assert abs(candidate(1.33) - 0.33) < 1e-6\n')
        completion = 'def solve(number):\n    return None\n'

        with self.assertRaises(AssertionError):
            self._exec_humaneval(completion, test)

    def test_humaneval_canonical_float_answer_passes(self):
        test = ('def check(candidate):\n'
                '    assert candidate(3.5) == 0.5\n'
                '    assert abs(candidate(1.33) - 0.33) < 1e-6\n')
        completion = ('def solve(number):\n'
                      '    return number - int(number)\n')

        self._exec_humaneval(completion, test)

    def test_humaneval_nested_forged_equality_is_rejected(self):
        test = 'def check(candidate):\n    assert candidate() == [1]\n'
        completion = ('def solve():\n'
                      '    class _Always(int):\n'
                      '        def __new__(cls):\n'
                      '            return int.__new__(cls, 999)\n'
                      '        def __eq__(self, other):\n'
                      '            return True\n'
                      '    return [_Always()]\n')

        with self.assertRaises(AssertionError):
            self._exec_humaneval(completion, test)

    def test_humaneval_prelude_matches_module_helpers(self):
        test = 'def check(candidate):\n    assert candidate() == 1\n'
        program = make_humaneval_test_type_aware(test)
        self.assertIn(TYPE_AWARE_EQUAL_NAME, program)
        self.assertIn(TYPE_AWARE_CLOSE_NAME, program)
        namespace = {}
        exec(program, namespace)
        self.assertFalse(namespace[TYPE_AWARE_EQUAL_NAME](_AlwaysEqual(), 1))
        self.assertTrue(namespace[TYPE_AWARE_EQUAL_NAME]([1, 2], [1, 2]))
        self.assertFalse(
            namespace[TYPE_AWARE_CLOSE_NAME](_AlwaysEqualArith(), 0.33, 1e-6))
        self.assertTrue(namespace[TYPE_AWARE_CLOSE_NAME](0.33, 0.33, 1e-6))

    def test_humaneval_invalid_test_is_left_unchanged(self):
        source = 'def check(:\n'
        self.assertEqual(make_humaneval_test_type_aware(source), source)
