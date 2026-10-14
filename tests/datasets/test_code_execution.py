import unittest

from opencompass.utils.code_execution import (TYPE_AWARE_EQUAL_NAME,
                                              make_assertions_type_aware,
                                              type_aware_equal)


class _AlwaysEqual:

    def __eq__(self, other):
        return True


class _AlwaysEqualInt(int):

    def __new__(cls):
        return int.__new__(cls, 999)

    def __eq__(self, other):
        return True


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
