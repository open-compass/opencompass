"""Tests for the FollowBench judge-output postprocessor."""

import ast
import re
from pathlib import Path
from types import FunctionType
from unittest.mock import Mock

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
@pytest.mark.parametrize(
    "prediction, level, expected",
    [
        ("['YES', 'NO']", 2, (0, 0.5)),
        ("['YES', 'YES']", 2, (1, 1)),
        ("['NO', 'PARTIAL']", 2, (0, 0)),
        ("['True', 'False']", 2, (0, 0.5)),
        ("['YES', 'MAYBE', 'UNKNOWN', 'N/A']", 4, (0, 0.25)),
        ("```\n['YES', 'NO']\n```", 2, (0, 0.5)),
    ],
)
def test_followbench_postprocessor_keeps_valid_label_lists(
        path: Path, prediction: str, level: int, expected: tuple) -> None:
    postprocessor = _load_postprocessor(path)

    assert postprocessor({
        "prediction": prediction,
        "gold": {
            "level": level
        }
    }) == expected


@pytest.mark.parametrize("path", POSTPROCESSOR_PATHS)
def test_followbench_postprocessor_rejects_expressions(path: Path) -> None:
    postprocessor = _load_postprocessor(path)
    record_call = Mock(return_value="YES")
    postprocessor.__globals__["record_call"] = record_call

    result = postprocessor({
        "prediction": "[record_call(), 'NO']",
        "gold": {
            "level": 2
        }
    })

    record_call.assert_not_called()
    assert result == (-1, -1)
