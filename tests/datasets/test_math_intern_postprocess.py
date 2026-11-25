"""Regression tests for math_intern.math_intern_postprocess.

The ``Extractor.extract_answer`` branch that handles predictions split on
``\\n\\nQuestion:`` used ``if pred.find('The answer is'):`` as a truthiness
test.  ``str.find`` returns ``0`` for a match at position 0 (falsy) and ``-1``
when the marker is absent (truthy), so the condition was inverted: a prediction
whose remaining text *starts* with ``The answer is`` was skipped and scored as
empty, while a prediction *without* the marker ran the slice anyway and
returned a garbage substring.  The fix makes the branch ``>= 0`` like every
sibling branch in the same method.

The real module cannot be imported directly here because
``opencompass.datasets.__init__`` eagerly pulls in heavy optional
dependencies (transformers, etc.).  Following the convention established by
``test_followbench_postprocess.py`` we load the ``Extractor`` class and the
``math_intern_postprocess`` function straight from the source file via AST so
the actual public postprocessor is exercised.
"""

import ast
from pathlib import Path
from types import FunctionType

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPOSITORY_ROOT / 'opencompass/datasets/math_intern.py'


def _load_postprocessor() -> FunctionType:
    tree = ast.parse(SOURCE_PATH.read_text(encoding='utf-8'),
                     filename=str(SOURCE_PATH))
    wanted = {'Extractor', 'math_intern_postprocess'}
    nodes = [
        node for node in tree.body if getattr(node, 'name', None) in wanted
    ]
    assert {
        getattr(n, 'name', None)
        for n in nodes
    } == wanted, (
        'expected Extractor class and math_intern_postprocess function in '
        f'{SOURCE_PATH}')
    # Strip the registry decorator so the function can be defined without the
    # opencompass registry import side-effects.
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    namespace = {}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(
        compile(module, filename=str(SOURCE_PATH), mode='exec'),  # noqa: S102
        namespace)
    return namespace['math_intern_postprocess']


POSTPROCESS = _load_postprocessor()


def test_answer_marker_at_position_zero_is_extracted():
    # After the ``\n\nQuestion:`` split the remaining text begins exactly with
    # ``The answer is`` so ``str.find`` returns 0.  The buggy ``if find(...):``
    # treated that 0 as falsy and dropped the correct answer, returning None.
    text = 'The answer is 42\n\nQuestion: what is next?'
    assert POSTPROCESS(text) == '42'


def test_marker_absent_after_question_split_returns_sentinel():
    # The ``\n\nQuestion:`` branch is entered but ``The answer is`` is absent,
    # so ``str.find`` returns -1.  The buggy truthiness test treated -1 as
    # truthy and sliced from index (-1 + len('The answer is')), yielding a
    # garbage substring instead of falling through to the final
    # ``return None``.
    text = 'The result works out to seven\n\nQuestion: what is next?'
    result = POSTPROCESS(text)
    assert result is None, (
        f'expected sentinel None when marker absent, got garbage {result!r}')


def test_common_final_answer_path_unchanged():
    # Counter-example / neighbour: the already-working ``The final answer is``
    # branch is untouched by the fix and must keep extracting the same value.
    text = 'Some reasoning. The final answer is $7$. done'
    assert POSTPROCESS(text) == '7'
