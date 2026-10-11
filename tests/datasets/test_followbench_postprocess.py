"""Tests for the FollowBench judge-output postprocessor."""

import ast
import re
from pathlib import Path
from types import FunctionType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POSTPROCESSOR_PATHS = (
    REPOSITORY_ROOT / "opencompass/datasets/subjective/followbench.py",
    REPOSITORY_ROOT / "opencompass/summarizers/subjective/followbench.py",
)


def _load_postprocessor(path: Path) -> FunctionType:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "post_process_followbench")
    namespace = {"ast": ast, "re": re}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, filename=str(path), mode="exec"),
         namespace)  # noqa: S102
    return namespace["post_process_followbench"]


@pytest.mark.parametrize("path", POSTPROCESSOR_PATHS)
def test_followbench_postprocessor_keeps_valid_label_lists(path: Path) -> None:
    postprocessor = _load_postprocessor(path)

    assert postprocessor({
        "prediction": "['YES', 'NO']",
        "gold": {
            "level": 2
        }
    }) == (0, 0.5)


@pytest.mark.parametrize("path", POSTPROCESSOR_PATHS)
def test_followbench_postprocessor_rejects_expressions(path: Path) -> None:
    postprocessor = _load_postprocessor(path)

    prediction = "[__import__('os').system('echo should-not-run')]"

    assert postprocessor({
        "prediction": prediction,
        "gold": {
            "level": 2
        }
    }) == (-1, -1)
