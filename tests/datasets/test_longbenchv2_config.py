"""Tests for LongBench v2 dataset configs."""

import ast
from pathlib import Path


def _longbenchv2_prompt():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root /
        'opencompass/configs/datasets/longbenchv2/longbenchv2_gen_75fbba.py'
    )
    tree = ast.parse(config_path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != 'prompt':
                continue
            if isinstance(keyword.value, ast.Constant):
                prompt = keyword.value.value
                if 'The correct answer is (insert answer here)' in prompt:
                    return prompt
    raise AssertionError('LongBench v2 prompt not found')


def test_longbenchv2_prompt_has_complete_answer_format():
    """Test LongBench v2 prompt keeps the official answer format."""
    assert _longbenchv2_prompt().endswith(
        'Format your response as follows: '
        '"The correct answer is (insert answer here)".')
