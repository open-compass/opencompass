import ast
import inspect
import textwrap
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

TYPE_AWARE_EQUAL_NAME = '__opencompass_type_aware_equal'
TYPE_AWARE_CLOSE_NAME = '__opencompass_type_aware_close'


def type_aware_equal(actual: Any, expected: Any) -> bool:
    """Compare values without accepting forged ``__eq__`` on return objects."""
    if type(actual) is not type(expected):
        return False

    if isinstance(actual, (list, tuple)):
        return (len(actual) == len(expected) and all(
            type_aware_equal(a, e) for a, e in zip(actual, expected)))

    if isinstance(actual, Mapping):
        if len(actual) != len(expected):
            return False

        matched_expected_keys = set()
        for actual_key, actual_value in actual.items():
            match_idx = None
            for idx, (expected_key,
                      expected_value) in enumerate(expected.items()):
                if idx in matched_expected_keys:
                    continue
                if (type_aware_equal(actual_key, expected_key)
                        and type_aware_equal(actual_value, expected_value)):
                    match_idx = idx
                    break
            if match_idx is None:
                return False
            matched_expected_keys.add(match_idx)
        return True

    if isinstance(actual, (set, frozenset)):
        if len(actual) != len(expected):
            return False

        expected_items = list(expected)
        matched_expected_items = set()
        for actual_item in actual:
            match_idx = None
            for idx, expected_item in enumerate(expected_items):
                if idx in matched_expected_items:
                    continue
                if type_aware_equal(actual_item, expected_item):
                    match_idx = idx
                    break
            if match_idx is None:
                return False
            matched_expected_items.add(match_idx)
        return True

    try:
        result = actual == expected
    except Exception:
        return False
    return type(result) is bool and result


def type_aware_close(actual: Any, expected: Any, tolerance: Any) -> bool:
    """Compare numbers within tolerance without a forged ``__sub__``."""
    if type(actual) is not int and type(actual) is not float:
        return False
    if type(expected) is not int and type(expected) is not float:
        return False
    if type(tolerance) is not int and type(tolerance) is not float:
        return False
    try:
        difference = abs(actual - expected)
    except Exception:
        return False
    if type(difference) is not int and type(difference) is not float:
        return False
    try:
        result = difference < tolerance
    except Exception:
        return False
    return type(result) is bool and result


class _TypeAwareAssertTransformer(ast.NodeTransformer):
    """Replace ``assert actual == expected`` with a type-aware comparison."""

    def visit_Assert(self, node: ast.Assert) -> ast.Assert:
        self.generic_visit(node)
        comparison = node.test
        if (isinstance(comparison, ast.Compare) and len(comparison.ops) == 1
                and isinstance(comparison.ops[0], ast.Eq)):
            node.test = ast.Call(
                func=ast.Name(id=TYPE_AWARE_EQUAL_NAME, ctx=ast.Load()),
                args=[comparison.left, comparison.comparators[0]],
                keywords=[],
            )
        return node


def make_assertions_type_aware(source: str) -> str:
    """Harden direct equality assertions in a dataset-provided test program."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    tree = _TypeAwareAssertTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class _TypeAwareToleranceTransformer(ast.NodeTransformer):
    """Replace absolute-difference asserts with a type-aware check."""

    def visit_Assert(self, node: ast.Assert) -> ast.Assert:
        self.generic_visit(node)
        comparison = node.test
        if not isinstance(comparison, ast.Compare):
            return node
        if len(comparison.ops) != 1 or not isinstance(comparison.ops[0],
                                                      ast.Lt):
            return node
        call = comparison.left
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == 'abs' and len(call.args) == 1
                and not call.keywords):
            return node
        difference = call.args[0]
        if not (isinstance(difference, ast.BinOp)
                and isinstance(difference.op, ast.Sub)):
            return node
        node.test = ast.Call(
            func=ast.Name(id=TYPE_AWARE_CLOSE_NAME, ctx=ast.Load()),
            args=[
                difference.left, difference.right, comparison.comparators[0]
            ],
            keywords=[],
        )
        return node


def _alias_function_source(fn: Any, public_name: str) -> str:
    """Return ``fn``'s source with its name replaced by ``public_name``."""
    source = textwrap.dedent(inspect.getsource(fn))
    return source.replace(fn.__name__, public_name)


@lru_cache(maxsize=1)
def _humaneval_helper_prelude() -> str:
    """Source of the helpers HumanEval must define inside the test program.

    ``human_eval`` runs the prompt, the completion and the test in one
    ``exec`` with empty globals, so the helpers cannot be injected through
    ``exec_globals`` the way MBPP and LCBench do.
    """
    return '\n'.join([
        'from collections.abc import Mapping',
        'from typing import Any',
        _alias_function_source(type_aware_equal, TYPE_AWARE_EQUAL_NAME),
        _alias_function_source(type_aware_close, TYPE_AWARE_CLOSE_NAME),
    ]) + '\n'


def make_humaneval_test_type_aware(source: str) -> str:
    """Harden a HumanEval test against forged equality and subtraction."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    tree = _TypeAwareAssertTransformer().visit(tree)
    tree = _TypeAwareToleranceTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return _humaneval_helper_prelude() + ast.unparse(tree) + '\n'
