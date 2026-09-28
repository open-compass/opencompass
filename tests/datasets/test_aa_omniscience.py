from unittest.mock import patch

import pytest
from datasets import Dataset

from opencompass.configs.datasets.aa_omniscience.aa_omniscience_llmjudge_rawprompt import (  # noqa: E501
    OMNISCIENCE_ANSWER_PROMPT, aa_omniscience_infer_cfg)
from opencompass.datasets.aa_omniscience import (
    AAOmniscienceDataset, _parse_aa_omniscience_judgement,
    aa_omniscience_llmjudge_postprocess)
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate


@patch('opencompass.datasets.aa_omniscience.load_dataset')
def test_aa_omniscience_loader_uses_huggingface_dataset(mock_load_dataset):
    expected = Dataset.from_list([{
        'domain': 'Science',
        'topic': 'Physics',
        'subtopic': 'Mechanics',
        'question_id': 1,
        'question': 'What is the answer?',
        'answer': '42',
    }])
    mock_load_dataset.return_value = expected

    actual = AAOmniscienceDataset.load(
        path='ArtificialAnalysis/AA-Omniscience-Public',
        cache_dir='/tmp/hf-cache',
    )

    mock_load_dataset.assert_called_once_with(
        path='ArtificialAnalysis/AA-Omniscience-Public',
        split='train',
        cache_dir='/tmp/hf-cache',
    )
    assert actual is expected


@pytest.mark.parametrize(
    ('response', 'expected'),
    [
        ('A', 'A'),
        ('A.', 'A'),
        ('B: INCORRECT', 'B'),
        ('PARTIAL_ANSWER', 'C'),
        ('not attempted', 'D'),
        ('A: CORRECT\nB: INCORRECT\nC: PARTIAL_ANSWER\n'
         'D: NOT_ATTEMPTED\nA', 'A'),
        ('The answer is CORRECT, not PARTIAL_ANSWER.\nA', 'A'),
        ('The answer may be CORRECT, but is ultimately INCORRECT.', 'B'),
        ('I cannot determine a grade.', None),
    ],
)
def test_parse_aa_omniscience_judgement(response, expected):
    assert _parse_aa_omniscience_judgement(response) == expected


def test_aa_omniscience_postprocess_uses_official_formulas():
    output = {
        '0': {
            'prediction': 'A'
        },
        '1': {
            'prediction': 'A: CORRECT'
        },
        '2': {
            'prediction': 'B'
        },
        '3': {
            'prediction': 'C: PARTIAL_ANSWER'
        },
        '4': {
            'prediction': 'D'
        },
    }

    result = aa_omniscience_llmjudge_postprocess(output, 'unused.json')

    assert result['score'] == pytest.approx(20)
    assert result['accuracy'] == pytest.approx(40)
    assert result['hallucination_rate'] == pytest.approx(100 / 3)
    assert result['attempt_rate'] == pytest.approx(80)
    assert result['correct_count'] == 2
    assert result['incorrect_count'] == 1
    assert result['partial_answer_count'] == 1
    assert result['not_attempted_count'] == 1
    assert result['total'] == 5
    assert result['details']['3']['grade_letter'] == 'C'


def test_aa_omniscience_postprocess_defaults_invalid_judge_to_not_attempted():
    result = aa_omniscience_llmjudge_postprocess(
        {'0': {
            'prediction': 'I cannot determine a grade.'
        }}, 'unused.json')

    assert result['score'] == 0
    assert result['not_attempted_count'] == 1
    assert result['parse_error_count'] == 1
    assert result['details']['0']['grade_letter'] == 'D'
    assert result['details']['0']['judge_parsed'] is None


def test_aa_omniscience_raw_prompt_matches_official_answer_prompt():
    cfg = aa_omniscience_infer_cfg['prompt_template']
    template = RawPromptTemplate(messages=cfg['messages'])

    messages = template.generate_item({
        'domain': 'Science',
        'subtopic': 'Physics',
        'question': 'What is the answer?',
    })

    assert messages == [
        {
            'role':
            'system',
            'content':
            OMNISCIENCE_ANSWER_PROMPT.format(domain='Science',
                                             subtopic='Physics'),
        },
        {
            'role': 'user',
            'content': 'What is the answer?',
        },
    ]
