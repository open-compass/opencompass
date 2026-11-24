import hashlib
import json
import math
from unittest.mock import patch

import pytest
from mmengine.config import Config

from opencompass.datasets.sumsurvey import (SUMSURVEY_PROMPT, SumSurveyDataset,
                                            SumSurveyEvaluator,
                                            aggregate_scores,
                                            build_judge_input,
                                            parse_judge_output, weighted_score)
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.utils.prompt import safe_format

CONFIG = 'opencompass/configs/datasets/SumSurvey/sumsurvey_llmjudge_gen.py'
# sha256 of ``system`` / ``rubric`` in judge_prompt_v1.1.json of SumSurvey
# Eval 1.1
OFFICIAL_PROMPT_SHA256 = {
    'system':
    '156e3145c058ab89581e7d8902a9f2221ebd096cf1e6494a8c432cd556382678',
    'rubric':
    'fc9c8830a76b1d451a0ccb90cc1db8bfd8b821c1214883b288a6c7aedad0a191',
}


def _sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _judgement(cov=3, prec=3, coh=3, nonred=3, **overrides):
    value = {
        'coverage': cov,
        'precision': prec,
        'coherence': coh,
        'non_redundancy': nonred,
        'missing_key_points': [],
        'unsupported_or_conflicting_claims': ['x'],
        'rationale': 'ok',
    }
    value.update(overrides)
    return value


def _write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows),
                    encoding='utf-8')


ROWS = [{
    'id': f'sumsurvey-test-000{i}',
    'prompt': f'Paper body {i} with {{braces}}.',
    'messages': [],
    'reference': f'Author abstract {i}.',
    'metadata': {
        'split': 'test',
        'line_number': i
    },
} for i in (1, 2)]


def test_loader_builds_paper_prompt(tmp_path):
    _write_jsonl(tmp_path / 'sumsurvey.jsonl', ROWS)
    ds = SumSurveyDataset.load(path=str(tmp_path))
    assert len(ds) == 2
    assert ds[0]['id'] == 'sumsurvey-test-0001'
    assert ds[0]['reference'] == 'Author abstract 1.'
    assert ds[0]['messages'] == [{
        'role':
        'user',
        'content': ('Please summarize the following contents with no more '
                    'than 256 words: Paper body 1 with {braces}.'),
    }]


def test_loader_resolves_path_with_get_data_path(tmp_path):
    path = tmp_path / 'sumsurvey.jsonl'
    _write_jsonl(path, ROWS[:1])
    with patch('opencompass.datasets.sumsurvey.get_data_path',
               return_value=str(path)) as mocked:
        ds = SumSurveyDataset.load()
    mocked.assert_called_once_with('opencompass/sumsurvey')
    assert len(ds) == 1


def test_judge_input_is_canonical_json():
    assert build_judge_input('Gold "abs"', 'Cand é') == (
        '{"author_abstract":"Gold \\"abs\\"","candidate_abstract":"Cand é"}')


def test_parse_judge_output_follows_v1_1_schema():
    assert parse_judge_output(json.dumps(_judgement())) == _judgement()
    fenced = '```json\n' + json.dumps(_judgement()) + '\n```'
    assert parse_judge_output(fenced) == _judgement()
    invalid = [
        json.dumps(_judgement(cov=3.0)),
        json.dumps(_judgement(cov=5)),
        json.dumps(_judgement(cov=True)),
        json.dumps(_judgement(extra=1)),
        json.dumps({
            k: v
            for k, v in _judgement().items() if k != 'rationale'
        }),
        json.dumps(_judgement(rationale=' ')),
        json.dumps(_judgement(missing_key_points=[''])),
        json.dumps(_judgement(missing_key_points=['a'] * 13)),
        'Here: ' + json.dumps(_judgement()),
        '[1, 2]',
        None,
    ]
    for text in invalid:
        assert parse_judge_output(text) is None, text


def test_weighted_score():
    assert weighted_score(_judgement(3, 3, 3, 3)) == pytest.approx(75.0)
    assert weighted_score(_judgement(4, 0, 0, 0)) == pytest.approx(45.0)
    assert weighted_score(_judgement(2, 1, 4, 3)) == pytest.approx(
        100 * (0.9 + 0.3 + 0.6 + 0.3) / 4)


def test_aggregate_excludes_failures_and_reports_ratio():
    statuses = ['succeeded', 'succeeded', 'source_failed', 'judge_failed']
    judgements = [_judgement(4, 4, 4, 4), _judgement(2, 2, 2, 2), None, None]
    result = aggregate_scores(statuses, judgements)
    assert result['score'] == pytest.approx(75.0)
    assert result['score_failures_as_zero'] == pytest.approx(37.5)
    assert result['scored_ratio'] == pytest.approx(0.5)
    assert result['num_scored'] == 2
    assert result['source_failed'] == 1
    assert result['judge_failed'] == 1
    assert result['dim_coverage'] == pytest.approx(3.0)

    empty = aggregate_scores(['source_failed'], [None])
    assert math.isnan(empty['score'])
    assert math.isnan(empty['dim_precision'])
    assert empty['score_failures_as_zero'] == 0


def _evaluator(tmp_path, max_attempts=5):
    evaluator = SumSurveyEvaluator(
        prompt_template=dict(type=RawPromptTemplate,
                             messages=[
                                 dict(role='user', content='{judge_input}'),
                             ]),
        judge_cfg=dict(type='unused'),
        max_attempts=max_attempts,
    )
    evaluator._out_dir = str(tmp_path / 'sumsurvey')
    return evaluator


def _test_set(tmp_path, n):
    rows = [dict(ROWS[0], id=f'id{i}', reference=f'REF{i}') for i in range(n)]
    _write_jsonl(tmp_path / 'sumsurvey.jsonl', rows)
    # built like the eval task does
    return SumSurveyDataset(path=str(tmp_path),
                            reader_cfg=dict(input_columns=['messages'],
                                            output_column='reference')).test


def test_evaluator_retries_and_skips_empty_outputs(tmp_path):
    # sample 0 parses at once, 1 on the retry, 2 never; 3 is empty
    replies = {
        'judge': [json.dumps(_judgement(4, 4, 4, 4)), 'oops', 'oops'],
        'judge_retry1': [json.dumps(_judgement(2, 2, 2, 2)), 'still bad'],
        'judge_retry2': ['still bad'],
    }
    calls = []

    def fake_judge(self, judge_inputs, name):
        calls.append((name, judge_inputs))
        return {
            str(j): {
                'prediction': reply
            }
            for j, reply in enumerate(replies[name])
        }

    test_set = _test_set(tmp_path, 4)
    evaluator = _evaluator(tmp_path, max_attempts=3)
    with patch.object(SumSurveyEvaluator, '_judge', fake_judge):
        result = evaluator.evaluate(k=1,
                                    n=1,
                                    original_dataset=test_set,
                                    predictions=['c0', 'c1', 'c2', '  '],
                                    references=test_set['reference'],
                                    test_set=test_set)

    assert [name
            for name, _ in calls] == ['judge', 'judge_retry1', 'judge_retry2']
    assert calls[0][1] == [
        build_judge_input(f'REF{i}', f'c{i}') for i in range(3)
    ]
    assert calls[1][1] == [
        build_judge_input('REF1', 'c1'),
        build_judge_input('REF2', 'c2')
    ]
    assert calls[2][1] == [build_judge_input('REF2', 'c2')]
    assert result['score'] == pytest.approx(75.0)
    assert result['scored_ratio'] == pytest.approx(0.5)
    assert result['source_failed'] == 1
    assert result['judge_failed'] == 1
    # evaluate() wraps every detail field in a per-replica list
    details = [{
        k: v[0]
        for k, v in d.items() if k != 'example_abbr'
    } for d in result['details']]
    assert [d['status'] for d in details
            ] == ['succeeded', 'succeeded', 'judge_failed', 'source_failed']
    assert [d['judge_attempts'] for d in details] == [1, 2, 3, 0]
    assert details[1]['dimensions']['coverage'] == 2


def test_config_matches_v1_1():
    (dataset, ) = Config.fromfile(CONFIG).sumsurvey_datasets
    assert dataset.path == 'opencompass/sumsurvey'

    template = RawPromptTemplate(**{
        k: v
        for k, v in dataset.infer_cfg.prompt_template.items() if k != 'type'
    })
    entry = {
        'messages': [{
            'role': 'user',
            'content': SUMSURVEY_PROMPT + 'Body {x}'
        }]
    }
    assert template.generate_item(entry) == entry['messages']

    evaluator = dataset.eval_cfg.evaluator
    assert evaluator.type is SumSurveyEvaluator
    system, user = evaluator.prompt_template.messages
    assert _sha256(system.content) == OFFICIAL_PROMPT_SHA256['system']
    judge_input = build_judge_input('GOLD', 'CAND')
    rendered = safe_format(user.content, judge_input=judge_input)
    rubric, tail = rendered.split('\n\nINPUT JSON:\n')
    assert _sha256(rubric) == OFFICIAL_PROMPT_SHA256['rubric']
    assert tail == judge_input
    assert evaluator.max_attempts == 5
    assert 'judge_cfg' in evaluator
