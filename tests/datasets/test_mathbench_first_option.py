"""Regression test for the missing comma in mathbench.first_option_postprocess.

Place at: tests/datasets/test_mathbench_first_option.py

This test loads ``first_option_postprocess`` via AST (no heavy opencompass /
mmengine imports), matching the convention already used by
tests/datasets/test_followbench_postprocess.py.

It fails on the buggy code (missing trailing comma at
opencompass/datasets/mathbench.py:168 fuses two patterns via implicit string
concatenation, destroying the standalone "只有选项X是对" pattern) and passes
once the comma is restored.
"""

import ast
import re
from pathlib import Path
from types import FunctionType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MATHBENCH_PATH = REPOSITORY_ROOT / "opencompass/datasets/mathbench.py"


def _load_first_option_postprocess() -> FunctionType:
    tree = ast.parse(MATHBENCH_PATH.read_text(encoding="utf-8"),
                     filename=str(MATHBENCH_PATH))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "first_option_postprocess")
    namespace = {"re": re}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, filename=str(MATHBENCH_PATH), mode="exec"),
         namespace)  # noqa: S102
    return namespace["first_option_postprocess"]


first_option_postprocess = _load_first_option_postprocess()


@pytest.mark.parametrize(
    "text, expected",
    [
        # A rejected distractor ("选项A错误") precedes the definitive
        # conclusion ("只有选项C是对" = "only option C is correct").
        # The dedicated 只有...是对 pattern must win so that C is returned;
        # with the missing comma it is destroyed and the loose cushion grabs
        # the first option letter (the distractor A).
        ("选项A错误，选项B也不对，只有选项C是对", "C"),
        ("选项D是干扰项，只有选项A是对", "A"),
        ("先看B，B不正确，只有选项D是对", "D"),
    ],
)
def test_only_option_is_correct_pattern(text, expected):
    answer, _pattern = first_option_postprocess(text, "ABCD")
    assert answer == expected


def test_standalone_guxuan_still_matches():
    # Sanity: the 故选 pattern that shared the fused line must remain intact.
    answer, _pattern = first_option_postprocess("故选 B", "ABCD")
    assert answer == "B"
