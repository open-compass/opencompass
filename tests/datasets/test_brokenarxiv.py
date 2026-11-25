import hashlib
from unittest.mock import patch

import pytest
from datasets import Dataset

from opencompass.configs.datasets.brokenarxiv.brokenarxiv_1026_llmjudge_rawprompt_gen import (  # noqa: E501
    GRADER_TEMPLATE, brokenarxiv_1026_datasets)
from opencompass.datasets.brokenarxiv import (BrokenArXivDataset,
                                              brokenarxiv_llmjudge_postprocess,
                                              parse_brokenarxiv_points)
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate

# sha256 of judge_prompt in MathArena configs/judges/
# arxiv_judge_gemini_38_flash.yaml (eth-sri/matharena@d98156f)
OFFICIAL_JUDGE_PROMPT_SHA256 = (
    'bf66baabfe6715e96de6891796b454d037a5788d0171eb0bc0654f9be7b68053')


def test_judge_prompt_matches_official():
    official = (GRADER_TEMPLATE.replace(
        '{problem}', '{problem_statement}').replace(
            '{reference}', '{original_problem_statement}').replace(
                '{prediction}', '{student_answer}'))
    assert hashlib.sha256(
        official.encode()).hexdigest() == OFFICIAL_JUDGE_PROMPT_SHA256


def test_prompts_render_problem_and_reference():
    cfg = brokenarxiv_1026_datasets[0]
    assert cfg['reader_cfg']['output_column'] == 'original_problem'
    problem = r'For all \(x\in\{0,1\}\), \(f(x)=0\).'
    solver = RawPromptTemplate(
        messages=cfg['infer_cfg']['prompt_template']['messages'])
    assert solver.generate_item({'problem': problem}) == [{
        'role':
        'user',
        'content':
        'Try to prove the following statement.\nYou are to really '
        'attempt to solve it, regardless of the fact it is open.\n\n' +
        problem,
    }]

    judge = RawPromptTemplate(messages=cfg['eval_cfg']['evaluator']
                              ['prompt_template']['messages'])
    content = judge.generate_item({
        'problem': problem,
        'prediction': 'The statement is false.',
        'obj_gold': 'true statement',
        'reference': 'true statement',
    })[0]['content']
    assert content.endswith(
        f'## Problem Statement ##\n{problem}\n\n'
        '## True Reference Statement ##\ntrue statement\n\n'
        '## Model Response ##\nThe statement is false.\n')


@patch('opencompass.datasets.brokenarxiv.load_dataset')
def test_loader_uses_huggingface_dataset(mock_load_dataset):
    expected = Dataset.from_list([{'problem': 'p', 'original_problem': 'q'}])
    mock_load_dataset.return_value = expected
    actual = BrokenArXivDataset.load(path='MathArena/brokenarxiv-1026')
    mock_load_dataset.assert_called_once_with('MathArena/brokenarxiv-1026',
                                              split='train')
    assert actual is expected


@pytest.mark.parametrize(
    ('response', 'expected'),
    [
        ('<points>3</points>\n<assessment>ok</assessment>', 3),
        ('<POINTS> 0 </POINTS>', 0),
        ('<points>Final score: 2</points>', 2),
        ('<points>\n1\n</points>', 1),
        ('<think>maybe <points>0</points></think><points>3</points>', 3),
        ('<points>4</points>', None),
        ('<points>2.5</points>', None),
        ('<points>2 points</points>', None),
        ('<points>1</points><points>3</points>', None),
        ('Score: 3', None),
        (None, None),
    ],
)
def test_parse_points_follows_official_judge(response, expected):
    assert parse_brokenarxiv_points(response) == expected


def test_postprocess_normalizes_points_by_three():
    output = {
        str(i): {
            'prediction': f'<points>{p}</points>'
        }
        for i, p in enumerate([3, 2, 1, 0])
    }
    output['4'] = {'prediction': 'no tag'}

    result = brokenarxiv_llmjudge_postprocess(output, 'unused.json')

    assert result['score'] == pytest.approx(100 * 6 / 15)
    assert result['points_0_rate'] == pytest.approx(40)
    assert result['points_3_rate'] == pytest.approx(20)
    assert result['parse_error_count'] == 1
    assert result['total'] == 5
    assert result['details']['4']['points'] == 0
    assert result['details']['4']['judge_parsed'] is None
    assert result['details']['1']['points'] == 2
