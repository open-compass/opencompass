import ast
from collections.abc import Mapping
from typing import Any

TYPE_AWARE_EQUAL_NAME = '__opencompass_type_aware_equal'


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
