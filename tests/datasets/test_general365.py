import json
from unittest.mock import patch

import pytest
from datasets import Dataset
from mmengine.config import Config

from opencompass.datasets.general365 import (
    General365Dataset,
    general365_llmjudge_postprocess,
    parse_general365_judgement,
)
from opencompass.evaluator import (CascadeEvaluator, GenericLLMEvaluator,
                                   MATHVerifyEvaluator)
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.summarizers import DefaultSummarizer


@patch('opencompass.datasets.general365.load_dataset')
def test_general365_loads_huggingface_and_filters_answer_types(mock_load):
    mock_load.return_value = Dataset.from_list([
        dict(id='1', question='Q1', answer='1', answer_type='number',
             float_round='0'),
        dict(id='2', question='Q2', answer='A', answer_type='single_choice',
             float_round='0'),
    ])

    dataset = General365Dataset.load(
        path='meituan-longcat/General365_Public',
        split='test',
        answer_types=['number', 'interval'],
    )

    mock_load.assert_called_once_with(
        path='meituan-longcat/General365_Public', split='test')
    assert len(dataset) == 1
    assert dataset[0]['answer_type'] == 'number'


def test_general365_official_judge_postprocessor():
    assert parse_general365_judgement('{"accuracy": true}') is True
    assert parse_general365_judgement(
        '```json\n{"accuracy": false}\n```') is False
    assert parse_general365_judgement('Correct') is None

    output = {
        '0': {
            'prediction': json.dumps({'accuracy': True})
        },
        '1': {
            'prediction': json.dumps({'accuracy': False})
        },
        '2': {
            'prediction': 'invalid'
        },
    }
    result = general365_llmjudge_postprocess(output, 'unused.json')
    assert result['accuracy'] == pytest.approx(100 / 3)
    assert result['correct_count'] == 1
    assert result['incorrect_count'] == 1
    assert result['parse_error_count'] == 1
    assert result['details']['0']['correct'] is True


def test_general365_config_uses_rawprompt_and_requested_evaluators():
    config = Config.fromfile(
        'opencompass/configs/datasets/General365/'
        'general365_rawprompt_cascade_llmjudge_gen.py')
    math_cfg, text_cfg = config.general365_datasets

    assert math_cfg.infer_cfg.prompt_template.type is RawPromptTemplate
    assert math_cfg.eval_cfg.evaluator.type is CascadeEvaluator
    assert math_cfg.eval_cfg.evaluator.rule_evaluator.type is (
        MATHVerifyEvaluator)
    assert math_cfg.eval_cfg.evaluator.llm_evaluator.type is (
        GenericLLMEvaluator)
    assert math_cfg.eval_cfg.evaluator.parallel is False
    assert text_cfg.eval_cfg.evaluator.type is GenericLLMEvaluator
    assert text_cfg.answer_types == [
        'text', 'single_choice', 'multiple_choice'
    ]


def test_general365_summary_group_uses_sample_count_weights():
    config = Config.fromfile(
        'opencompass/configs/summarizers/groups/General365.py')
    group = config.general365_summary_groups[0]

    assert group.name == 'General365'
    assert group.weights == {
        'General365-mathverify': 484,
        'General365-text': 236,
    }

    summarizer = DefaultSummarizer.__new__(DefaultSummarizer)
    summarizer.summary_groups = [group]
    summarizer.model_abbrs = ['test-model']
    raw_results = {
        'test-model': {
            'General365-mathverify': {
                'accuracy': 75.0
            },
            'General365-text': {
                'accuracy': 50.0
            },
        }
    }
    parsed_results = {
        'test-model': {
            'General365-mathverify': {
                'accuracy': 75.0
            },
            'General365-text': {
                'accuracy': 50.0
            },
        }
    }
    dataset_metrics = {
        'General365-mathverify': ['accuracy'],
        'General365-text': ['accuracy'],
    }
    dataset_eval_mode = {
        'General365-mathverify': 'gen',
        'General365-text': 'gen',
    }

    _, parsed_results, dataset_metrics, _ = (
        summarizer._calculate_group_metrics(
            raw_results, parsed_results, dataset_metrics, dataset_eval_mode))

    expected = (75.0 * 484 + 50.0 * 236) / 720
    assert parsed_results['test-model']['General365'][
        'weighted_average'] == pytest.approx(expected)
    assert dataset_metrics['General365'] == ['accuracy', 'weighted_average']
