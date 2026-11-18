"""Regression tests for tabmwp.extract_prediction case-sensitivity fix.

``extract_prediction`` matched the answer marker with the character class
``[Th]he`` (matches ``The`` or ``hhe``, *not* lowercase ``the``) in regexes
compiled without ``re.IGNORECASE``.  A model answering ``the answer is B``
therefore missed every letter-extraction pattern and fell through to the fuzzy
``score_string_similarity`` fallback, which can pick the wrong option and
miscompute accuracy.  The fix changes the class to ``[Tt]he`` so both ``The``
and ``the`` match.

``opencompass.datasets.__init__`` eagerly imports heavy optional dependencies,
so -- following the ``tests/datasets/test_followbench_postprocess.py``
convention -- we load ``extract_prediction`` (and its helper
``score_string_similarity``) straight from the source file via AST and call the
actual function.
"""

import ast
import random
import re
from pathlib import Path
from types import FunctionType

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPOSITORY_ROOT / 'opencompass/datasets/tabmwp.py'


def _load_extract_prediction() -> FunctionType:
    tree = ast.parse(SOURCE_PATH.read_text(encoding='utf-8'),
                     filename=str(SOURCE_PATH))
    wanted = {'extract_prediction', 'score_string_similarity'}
    nodes = [n for n in tree.body if getattr(n, 'name', None) in wanted]
    assert {
        getattr(n, 'name', None)
        for n in nodes
    } == wanted, ('expected extract_prediction and score_string_similarity in '
                  f'{SOURCE_PATH}')
    namespace = {'re': re, 'random': random, 'np': np}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(
        compile(module, filename=str(SOURCE_PATH), mode='exec'),  # noqa: S102
        namespace)
    return namespace['extract_prediction']


EXTRACT = _load_extract_prediction()

# Option B is the correct choice; its text is 'blue sky'.  The distractor
# 'red apple' shares more whitespace-split tokens with the lowercase sentence,
# so the fuzzy fallback (reached only when the marker pattern misses) selects
# it instead -- the exact miscomputation the fix prevents.
OPTIONS = ['red apple', 'blue sky']


def test_lowercase_the_answer_is_selects_correct_option():
    # Buggy [Th]he never matched lowercase 'the', so this fell through to the
    # fuzzy fallback and returned 'red apple'.  With [Tt]he it matches and
    # returns option B -> 'blue sky'.
    assert EXTRACT('I think the answer is B', options=OPTIONS) == 'blue sky'


def test_uppercase_the_answer_is_unchanged():
    # Counter-example / neighbour: the already-working uppercase 'The' case is
    # unaffected by the fix and must still resolve to option B.
    assert EXTRACT('The answer is B', options=OPTIONS) == 'blue sky'
