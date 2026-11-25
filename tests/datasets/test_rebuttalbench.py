import hashlib
import json
import math
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset
from mmengine.config import Config

from opencompass.datasets.rebuttalbench import (CATEGORIES,
                                                RebuttalBenchDataset,
                                                RebuttalBenchEvaluator,
                                                aggregate_scores,
                                                build_judge_input,
                                                dimension_scores,
                                                parse_judge_output)
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.utils.prompt import safe_format

CONFIG_DIR = 'opencompass/configs/datasets/RebuttalBench/'
# sha256 of SYS4 / SYS5 in eval/agent_prompt.py of
# https://github.com/Zhitao-He/RebuttalAgent
OFFICIAL_PROMPT_SHA256 = {
    'judge':
    'dea5073823cff1ebd854933c596065e094b3371aba3a9875a3040338061a6750',
    'base': '627d42990fba127347132b5275c603311d23d16f1263dfe053a8ee6c1c6e856e',
}


def _sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


RAR_ROW = {
    'content': 'The whole review body.',
    'rate': '6',
    'paper_id': 'abc123',
    'global_profile': {
        'overall_stance': 'Probably Reject'
    },
    'comment': {
        'category': 'Experimental Rigor',
        'comment_id': 5,
        'comment_text': 'Ablations are only in the appendix.',
        'confidence': 10,
        'severity': 'Major',
        'sub_category': 'Ablation/Analysis Missing',
    },
    'original_index': 0,
    'top5_text': ['fragment one', 'fragment two', 'fragment three'],
    'top5_value': [0.9, 0.8, 0.7],
}

R2_ROW = {
    'reviewer_id': 'r1',
    'review_content': 'Review body from R2.',
    'conference_year_track': 'ICLR 2023 Poster',
    'paper_id': 'def456',
    'global_profile': {},
    'comment': {
        'category': 'Presentation & Clarity',
        'comment_id': 2,
        'comment_text': 'Notation is unclear.',
        'confidence': 8,
        'severity': 'Minor',
        'sub_category': 'Unclear Notation',
    },
    'original_index': 1,
    'top5_text': ['r2 fragment'],
    'top5_value': [0.5],
}

NO_FRAGMENT_ROW = dict(RAR_ROW, paper_id='nofrag', top5_text=[], top5_value=[])


def _judge_json(a, c, p, co):
    return json.dumps({
        'score': {
            'Attitude': a,
            'Clarity': c,
            'Persuasiveness': p,
            'Constructiveness': co,
        },
        'score_explanation': 'ok',
    })


def test_loader_handles_both_schemas_and_drops_rows_without_fragment(tmp_path):
    path = tmp_path / 'RAR_test.json'
    path.write_text(json.dumps([RAR_ROW, NO_FRAGMENT_ROW, R2_ROW]),
                    encoding='utf-8')

    ds = RebuttalBenchDataset.load(path=str(tmp_path), split='RAR_test')

    assert len(ds) == 2
    assert ds[0]['review_content'] == 'The whole review body.'
    assert ds[0]['paper_fragment'] == 'fragment one'
    assert ds[0]['category'] == 'Experimental Rigor'
    assert ds[1]['review_content'] == 'Review body from R2.'
    assert ds[1]['paper_id'] == 'def456'
    assert ds[1]['comment_id'] == 2


def test_loader_resolves_path_with_get_data_path(tmp_path):
    (tmp_path / 'R2_test.json').write_text(json.dumps([R2_ROW]),
                                           encoding='utf-8')
    with patch('opencompass.datasets.rebuttalbench.get_data_path',
               return_value=str(tmp_path)) as mocked:
        ds = RebuttalBenchDataset.load(split='R2_test')
    mocked.assert_called_once_with('opencompass/rebuttalbench')
    assert ds[0]['comment_text'] == 'Notation is unclear.'


def test_loader_rejects_unknown_split():
    with pytest.raises(ValueError):
        RebuttalBenchDataset.load(split='train')


def test_judge_input_matches_official_rw_input1():
    row = {
        'review_content': 'REVIEW',
        'comment_text': 'COMMENT',
        'paper_fragment': 'FRAGMENT',
    }
    assert build_judge_input(
        row, 'RESPONSE') == ('The whole review content is\nREVIEW\n'
                             'The comment content is:\nCOMMENT\n'
                             'The original response fragment is\nRESPONSE\n'
                             'The best paper fragment is: FRAGMENT\n')


def test_parse_judge_output_follows_official_extract1():
    assert parse_judge_output(_judge_json(8, 7, 6, 9))['score'] == {
        'Attitude': 8,
        'Clarity': 7,
        'Persuasiveness': 6,
        'Constructiveness': 9,
    }
    fenced = '```json\n' + _judge_json(8, 7, 6, 9) + '\n```'
    assert parse_judge_output(fenced) is not None
    # bare ``` fences and surrounding prose do not parse officially
    assert parse_judge_output('```\n' + _judge_json(1, 1, 1, 1) +
                              '\n```') is None
    assert parse_judge_output('Here: ' + _judge_json(1, 1, 1, 1)) is None
    assert parse_judge_output('[1, 2]') is None
    assert parse_judge_output(None) is None


def test_dimension_scores_requires_all_four_dimensions():
    assert dimension_scores(json.loads(_judge_json(8, 7, 6, 9))) == {
        'Attitude': 8.0,
        'Clarity': 7.0,
        'Persuasiveness': 6.0,
        'Constructiveness': 9.0,
    }
    assert dimension_scores({'score': {'Attitude': 8}}) is None
    assert dimension_scores({}) is None
    assert dimension_scores(None) is None


@pytest.mark.parametrize('bad', [
    -1, 99, True, False, 'NaN', 'Infinity', '-Infinity', None, 'unrated',
    pytest.param(10**400, id='overflow')
])
def test_invalid_dimension_excludes_whole_judgement(bad):
    valid = json.loads(_judge_json(0, '8', 6.5, 10))
    invalid = json.loads(_judge_json(8, 8, bad, 8))
    result = aggregate_scores([CATEGORIES[0]] * 2, [invalid, valid])

    assert dimension_scores(invalid) is None
    assert result['num_scored'] == 1
    assert result['score'] == pytest.approx((0 + 8 + 6.5 + 10) / 4)


def test_aggregate_scores_matches_ag_avg():
    categories = [
        'Experimental Rigor',
        'Experimental Rigor',
        'Presentation & Clarity',
        'Meta-Critique & Reviewer Behavior',  # not a target category
        'Novelty & Significance',  # judge output unusable
    ]
    judgements = [
        json.loads(_judge_json(8, 8, 8, 8)),
        json.loads(_judge_json(6, 4, 2, 0)),
        json.loads(_judge_json(10, 9, 8, 7)),
        json.loads(_judge_json(0, 0, 0, 0)),
        None,
    ]
    result = aggregate_scores(categories, judgements)

    assert result['num_scored'] == 3
    assert result['Attitude'] == pytest.approx(8.0)
    assert result['Constructiveness'] == pytest.approx(5.0)
    assert result['Experimental Rigor/Clarity'] == pytest.approx(6.0)
    assert result['Presentation & Clarity/Attitude'] == pytest.approx(10.0)
    assert math.isnan(result['Novelty & Significance/Attitude'])
    # overall_mode=sample: mean of the 12 (sample, dimension) values
    assert result['score'] == pytest.approx((32 + 12 + 34) / 12)
    # overall_mode=category: mean of the per-category means
    assert result['score_category_avg'] == pytest.approx((5.5 + 8.5) / 2)
    # without Attitude
    assert result['score_CPCo'] == pytest.approx((24 + 6 + 24) / 9)
    assert len([k for k in result if '/' in k]) == 4 * len(CATEGORIES)


def _evaluator(tmp_path, max_attempts=5):
    evaluator = RebuttalBenchEvaluator(
        prompt_template=dict(type=RawPromptTemplate,
                             messages=[
                                 dict(role='user', content='{judge_input}'),
                             ]),
        judge_cfg=dict(type='unused'),
        max_attempts=max_attempts,
    )
    evaluator._out_dir = str(tmp_path / 'rebuttalbench')
    return evaluator


def _test_set():
    return Dataset.from_list([{
        'review_content': f'REVIEW{i}',
        'comment_text': f'COMMENT{i}',
        'paper_fragment': f'FRAGMENT{i}',
        'category': category,
        'paper_id': f'p{i}',
        'comment_id': i,
    } for i, category in enumerate([
        'Experimental Rigor', 'Methodological Soundness', 'Experimental Rigor'
    ])])


def test_evaluator_retries_unparsable_judgements(tmp_path):
    # sample 1 parses on the second call, sample 2 never parses
    replies = {
        'judge': [_judge_json(8, 8, 8, 8), 'oops', 'oops'],
        'judge_retry1': [_judge_json(6, 6, 6, 6), 'still bad'],
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

    evaluator = _evaluator(tmp_path, max_attempts=3)
    with patch.object(RebuttalBenchEvaluator, '_judge', fake_judge):
        result = evaluator.score(['r0', 'r1', 'r2'], test_set=_test_set())

    assert [name
            for name, _ in calls] == ['judge', 'judge_retry1', 'judge_retry2']
    assert len(calls[1][1]) == 2 and 'COMMENT1' in calls[1][1][0]
    assert result['judge_parse_failures'] == 1
    assert result['num_scored'] == 2
    assert result['score'] == pytest.approx(7.0)
    assert result['Methodological Soundness/Attitude'] == pytest.approx(6.0)
    details = result['details']
    assert [d['judge_attempts'] for d in details] == [1, 2, 3]
    assert [d['counted'] for d in details] == [True, True, False]


@pytest.mark.parametrize('unusable', ['{}', _judge_json(8, 8, 99, 8)],
                         ids=['missing-scores', 'out-of-range'])
def test_retry_invalid_scores_with_existing_judge_files(tmp_path, unusable):
    evaluator = _evaluator(tmp_path, max_attempts=2)
    initial = {
        '0': {
            'prediction': _judge_json(8, 8, 8, 8)
        },
        '1': {
            'prediction': unusable
        },
        '2': {
            'prediction': 'not JSON'
        },
    }
    (tmp_path / 'rebuttalbench_replica0_judge.json').write_text(
        json.dumps(initial), encoding='utf-8')
    # The old retry batch contained only sample 2, at position 0.
    legacy = {'0': {'prediction': _judge_json(10, 10, 10, 10)}}
    (tmp_path / 'rebuttalbench_replica0_judge_retry1.json').write_text(
        json.dumps(legacy), encoding='utf-8')

    test_set = _test_set()
    predictions = ['r0', 'r1', 'r2']
    recovered = _judge_json(6, 6, 6, 6)
    exhausted = _judge_json(99, 99, 99, 99)
    retry_replies = {
        build_judge_input(test_set[1], predictions[1]): recovered,
        build_judge_input(test_set[2], predictions[2]): exhausted,
    }

    def generate(prompts, **kwargs):
        return [retry_replies.pop(chat[0]['content']) for chat in prompts]

    model = Mock(is_api=True, generation_kwargs={})
    model.parse_template.side_effect = lambda prompts, mode: prompts
    model.generate_from_template.side_effect = generate
    with patch('opencompass.datasets.rebuttalbench.build_model_from_cfg',
               return_value=model):
        result = evaluator.score(predictions, test_set=test_set)

    details = result['details']
    assert [d['judge_attempts'] for d in details] == [1, 2, 2]
    assert [d['counted'] for d in details] == [True, True, False]
    assert [d['judge_output'] for d in details
            ] == [initial['0']['prediction'], recovered, exhausted]
    assert details[2]['scores'] is None
    assert result['judge_parse_failures'] == 1
    assert result['num_scored'] == 2
    assert result['score'] == pytest.approx(7)
    assert not retry_replies

    # A second run reads the new retry file with the same row mapping.
    with patch('opencompass.datasets.rebuttalbench.build_model_from_cfg',
               side_effect=AssertionError('unexpected judge request')):
        resumed = evaluator.score(predictions, test_set=test_set)
    assert resumed['details'] == details


def test_evaluate_judges_raw_outputs_on_a_built_dataset(tmp_path):
    seen = []

    def fake_judge(self, judge_inputs, name):
        seen.extend(judge_inputs)
        return {
            str(j): {
                'prediction': _judge_json(9, 9, 9, 9)
            }
            for j in range(len(judge_inputs))
        }

    # built like the eval task does, which adds the subdivision/idx columns
    rows = [
        dict(RAR_ROW, comment=dict(RAR_ROW['comment'], comment_id=i))
        for i in range(3)
    ]
    (tmp_path / 'RAR_test.json').write_text(json.dumps(rows), encoding='utf-8')
    test_set = RebuttalBenchDataset(path=str(tmp_path),
                                    split='RAR_test',
                                    reader_cfg=dict(input_columns=[
                                        'review_content', 'comment_text',
                                        'paper_fragment'
                                    ],
                                                    output_column=None)).test
    evaluator = _evaluator(tmp_path)
    preds = [f'Thank you for the comment. R{i}' for i in range(3)]
    with patch.object(RebuttalBenchEvaluator, '_judge', fake_judge):
        result = evaluator.evaluate(k=1,
                                    n=1,
                                    original_dataset=test_set,
                                    predictions=preds,
                                    references=None,
                                    test_set=test_set)

    assert ('The original response fragment is\n'
            'Thank you for the comment. R0\n') in seen[0]
    assert result['score'] == pytest.approx(9.0)


@pytest.mark.parametrize('name, split', [
    ('rebuttalbench_rar_test_base_llmjudge_gen.py', 'RAR_test'),
    ('rebuttalbench_r2_test_base_llmjudge_gen.py', 'R2_test'),
])
def test_configs(name, split):
    (dataset, ) = Config.fromfile(CONFIG_DIR + name).rebuttalbench_datasets
    assert dataset.split == split
    assert dataset.path == 'opencompass/rebuttalbench'
    system, user = dataset.infer_cfg.prompt_template.messages
    assert _sha256(system.content) == OFFICIAL_PROMPT_SHA256['base']
    # R_input of ag_eval.py
    row = dict(review_content='REVIEW',
               comment_text='COMMENT',
               paper_fragment='FRAGMENT')
    assert safe_format(user.content,
                       **row) == ('The whole review content is\nREVIEW\n'
                                  'The target comment content is:\nCOMMENT\n'
                                  'The best paper fragment is: FRAGMENT\n')
    evaluator = dataset.eval_cfg.evaluator
    assert evaluator.type is RebuttalBenchEvaluator
    judge_system, judge_user = evaluator.prompt_template.messages
    assert _sha256(judge_system.content) == OFFICIAL_PROMPT_SHA256['judge']
    assert judge_user.content == '{judge_input}'
    assert evaluator.max_attempts == 5
    assert 'judge_cfg' in evaluator
    assert 'pred_postprocessor' not in evaluator
