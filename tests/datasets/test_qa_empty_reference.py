"""Regression tests for empty-reference handling in the substring-match QA
evaluators (``TriviaQAEvaluator`` and ``NQEvaluator``).

Both evaluators normalize every gold answer with ``general_postprocess`` and
then score a prediction correct with ``any(cand in pred for cand in cand_ans)``.
``general_postprocess`` can reduce a non-empty reference (e.g. a bare article
such as ``"A"``/``"An"``/``"The"``) to the empty string, and ``'' in pred`` is
always ``True`` in Python -- so a single empty reference would mark *every*
prediction correct and silently inflate the score.

The evaluators and ``general_postprocess`` are AST-loaded so the test does not
depend on heavy runtime imports (mmengine, datasets, ...), mirroring
``tests/datasets/test_followbench_postprocess.py``.
"""

import ast
import re
from pathlib import Path
from types import FunctionType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEXT_POSTPROCESSORS_PATH = (REPOSITORY_ROOT /
                            'opencompass/utils/text_postprocessors.py')
EVALUATOR_PATHS = {
    'TriviaQAEvaluator': REPOSITORY_ROOT / 'opencompass/datasets/triviaqa.py',
    'NQEvaluator': REPOSITORY_ROOT / 'opencompass/datasets/natural_question.py',
}


def _load_general_postprocess() -> FunctionType:
    tree = ast.parse(TEXT_POSTPROCESSORS_PATH.read_text(encoding='utf-8'),
                     filename=str(TEXT_POSTPROCESSORS_PATH))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'general_postprocess')
    function.decorator_list = []  # drop @TEXT_POSTPROCESSORS.register_module
    namespace = {'re': re}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, filename=str(TEXT_POSTPROCESSORS_PATH), mode='exec'),
         namespace)  # noqa: S102
    return namespace['general_postprocess']


def _load_score(class_name: str, path: Path) -> FunctionType:
    """AST-load the ``score`` method of an evaluator as a plain function.

    The method body only relies on ``general_postprocess`` and builtins, so we
    rebuild it as a standalone function taking ``(predictions, references)``.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'score')
    # Reshape ``def score(self, predictions, references)`` into a free
    # function ``def score(predictions, references)``.
    method.args.args = [a for a in method.args.args if a.arg != 'self']
    method.decorator_list = []
    namespace = {'general_postprocess': _load_general_postprocess()}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, filename=str(path), mode='exec'),
         namespace)  # noqa: S102
    return namespace['score']


@pytest.fixture(scope='module')
def general_postprocess():
    return _load_general_postprocess()


@pytest.mark.parametrize('text', ['A', 'An', 'The', 'the', 'a', 'A.'])
def test_general_postprocess_keeps_content_non_empty(general_postprocess,
                                                     text):
    # Documents the root cause: a non-empty, content-bearing input normalizes
    # to the empty string. (Kept as an xfail-style assertion of current
    # behavior so the evaluator-level guard is the thing actually enforced.)
    assert general_postprocess(text) == ''


@pytest.mark.parametrize('class_name, path', list(EVALUATOR_PATHS.items()))
def test_empty_reference_does_not_mark_wrong_prediction_correct(
        class_name, path):
    score = _load_score(class_name, path)
    # Gold aliases for a single example: a real answer plus an alias ('The')
    # that normalizes to ''. The model's answer is completely unrelated.
    result = score(['definitely not the answer'], [['Nikola Tesla', 'The']])
    assert result['score'] == 0.0, (
        f'{class_name}: an empty normalized reference must not mark a wrong '
        'prediction correct')
    assert result['details'][0]['correct'] is False


@pytest.mark.parametrize('class_name, path', list(EVALUATOR_PATHS.items()))
def test_all_references_empty_wrong_prediction_stays_wrong(class_name, path):
    score = _load_score(class_name, path)
    # Every alias collapses to '' -> must not be a free pass.
    result = score(['some unrelated text'], [['A', 'An', 'the']])
    assert result['score'] == 0.0
    assert result['details'][0]['correct'] is False


@pytest.mark.parametrize('class_name, path', list(EVALUATOR_PATHS.items()))
def test_correct_prediction_still_scores(class_name, path):
    score = _load_score(class_name, path)
    # The guard must not break genuine matches.
    result = score(['The answer is Paris.'], [['Paris']])
    assert result['score'] == 100.0
    assert result['details'][0]['correct'] is True
