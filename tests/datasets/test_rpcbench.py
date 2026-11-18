import json
from unittest.mock import patch

import mmengine
import pytest

from opencompass.datasets.rpcbench import (RPCBenchDataset,
                                           RPCBenchLLMEvaluator, parse_rating)
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate

PAPER = {
    'id': 'p1',
    'title': 'A Paper',
    'abstract': 'An abstract.',
    'qa_pairs': [
        {
            'question': 'How is $E_{content}$ defined?',
            'answer': 'As a loss.',
            'category': 'Method_Mechanics',
        },
        {
            'question': 'The method uses a loss.',
            'answer': 'True',
            'category': 'Claim_Verification',
        },
        {
            'question': 'Why this setup?',
            'answer': 'For speed.',
            'category': 'Experimental_Setup',
        },
    ],
}


@pytest.fixture
def repo(tmp_path):
    (tmp_path / 'split_metadata').mkdir()
    (tmp_path / 'split_metadata' / 'test.json').write_text(
        json.dumps(PAPER) + '\n', encoding='utf-8')
    (tmp_path / 'md' / 'test' / 'p1').mkdir(parents=True)
    (tmp_path / 'md' / 'test' / 'p1' / 'p1.md').write_text('# Paper body',
                                                           encoding='utf-8')
    return tmp_path


def test_loader_splits_open_and_claim_verification(repo):
    open_ds = RPCBenchDataset.load(path=str(repo), subset='open')
    cv_ds = RPCBenchDataset.load(path=str(repo), subset='cv')

    assert open_ds['part_idx'] == [1, 3]
    assert open_ds['category'] == ['Method_Mechanics', 'Experimental_Setup']
    assert cv_ds['part_idx'] == [2]
    assert cv_ds[0]['answer'] == 'True'
    assert open_ds[0]['paper_content'] == '# Paper body'
    assert open_ds[0]['title'] == 'A Paper'


@pytest.mark.parametrize('judgement, expected', [
    ('```json\n{"Correctness": {"rating": "4.15", "rationale": "ok"}}\n```',
     4.15),
    ('{"Correctness": {"rating": "3.20", "rationale": "uses \\alpha"}}', 3.2),
    ('<think>{"Correctness": {"rating": "1"}}</think>'
     '{"Correctness": {"rating": "2.50"}}', 2.5),
    ('Correctness rating: "rating": "3.75" (truncated', 3.75),
    ('{"Correctness": {"rating": "7.00"}}', 5.0),
    ('no score here', None),
    ('', None),
])
def test_parse_rating(judgement, expected):
    assert parse_rating(judgement, 'Correctness') == expected


class _FakeInferencer:
    """Write judge replies whose rating depends on judge, dimension and
    prediction."""

    ratings = {}

    def __init__(self, model, output_json_filepath, output_json_filename,
                 **kwargs):
        self.model = model
        self.path = f'{output_json_filepath}/{output_json_filename}'

    def inference(self, retriever, prompt_template):
        out = {}
        for i in range(len(retriever.test_ds)):
            system, user = retriever.generate_prompt_for_generate_task(
                i, '', prompt_template=prompt_template)
            dim, pred = system['content'], user['content']
            rating = self.ratings[(self.model, dim, pred)]
            out[str(i)] = {
                'prediction':
                f'```json\n{{"{dim}": {{"rating": "{rating}"}}}}\n```'
            }
        mmengine.dump(out, self.path)


def _evaluator(repo, tmp_path, judge_cfg):
    templates = {
        dim:
        dict(type=RawPromptTemplate,
             messages=[{
                 'role': 'system',
                 'content': dim
             }, {
                 'role': 'user',
                 'content': '{prediction}'
             }])
        for dim in RPCBenchLLMEvaluator.DIMENSIONS
    }
    evaluator = RPCBenchLLMEvaluator(
        prompt_templates=templates,
        judge_cfg=judge_cfg,
        dataset_cfg=dict(type=RPCBenchDataset,
                         path=str(repo),
                         subset='open',
                         reader_cfg=dict(input_columns=['question'],
                                         output_column='answer')),
    )
    evaluator._out_dir = str(tmp_path / 'judge' / 'rpcbench_test_open')
    return evaluator


@patch('opencompass.datasets.rpcbench.build_model_from_cfg',
       side_effect=lambda model_cfg: model_cfg['abbr'])
@patch('opencompass.datasets.rpcbench.GenInferencer', _FakeInferencer)
def test_evaluator_averages_judges_like_official(_, repo, tmp_path):
    preds = ['pred-a', 'pred-b']
    _FakeInferencer.ratings = {
        # judge A: sample 0 / sample 1
        ('A', 'Correctness', 'pred-a'): 4.0,
        ('A', 'Correctness', 'pred-b'): 2.0,
        ('A', 'Completeness', 'pred-a'): 3.0,
        ('A', 'Completeness', 'pred-b'): 1.0,
        ('A', 'Conciseness', 'pred-a'): 5.0,
        ('A', 'Conciseness', 'pred-b'): 3.0,
        # judge B
        ('B', 'Correctness', 'pred-a'): 5.0,
        ('B', 'Correctness', 'pred-b'): 5.0,
        ('B', 'Completeness', 'pred-a'): 4.0,
        ('B', 'Completeness', 'pred-b'): 2.0,
        ('B', 'Conciseness', 'pred-a'): 2.0,
        ('B', 'Conciseness', 'pred-b'): 2.0,
    }
    evaluator = _evaluator(
        repo, tmp_path,
        dict(judgers=[dict(abbr='A', path='a'),
                      dict(abbr='B', path='b')]))
    results = evaluator.score(preds, ['As a loss.', 'For speed.'])

    # per-judge means: A corr 3, comp 2, conc 4; B corr 5, comp 3, conc 2
    corr, comp, conc = 4.0, 2.5, 3.0
    f1 = 2 * corr * comp / (corr + comp)
    assert results['correctness'] == pytest.approx(corr * 20)
    assert results['completeness'] == pytest.approx(comp * 20)
    assert results['conciseness'] == pytest.approx(conc * 20)
    assert results['f1_like'] == pytest.approx(f1 * 20)
    assert results['informativeness'] == pytest.approx(conc * f1 * 4)
    assert results['A/correctness'] == pytest.approx(60)
    assert results['B/completeness'] == pytest.approx(60)
    # sample 1 only: corr (2+5)/2, comp (1+2)/2
    assert results['Experimental_Setup/correctness'] == pytest.approx(70)
    assert results['Experimental_Setup/completeness'] == pytest.approx(30)
    assert results['judge_parse_failures'] == 0
    assert results['details'][1]['ratings']['B']['Correctness'] == 5.0
    assert (tmp_path / 'judge' /
            'rpcbench_test_open_replica0_A_correctness.json').exists()


@patch('opencompass.datasets.rpcbench.GenInferencer')
@patch('opencompass.datasets.rpcbench.build_model_from_cfg')
def test_evaluator_reuses_existing_judge_outputs(mock_build, mock_infer, repo,
                                                 tmp_path):
    evaluator = _evaluator(repo, tmp_path, dict(abbr='J', path='j'))
    (tmp_path / 'judge').mkdir()
    for dim in RPCBenchLLMEvaluator.DIMENSIONS:
        mmengine.dump(
            {
                '0': {
                    'prediction': f'{{"{dim}": {{"rating": "5.00"}}}}'
                },
                '1': {
                    'prediction': 'unparsable'
                },
            }, tmp_path / 'judge' /
            f'rpcbench_test_open_replica0_{dim.lower()}.json')

    results = evaluator.score(['x', 'y'], ['As a loss.', 'For speed.'])

    mock_build.assert_not_called()
    mock_infer.assert_not_called()
    # unparsable ratings count as 0 like the official get_LLM_score
    assert results['correctness'] == pytest.approx(50)
    assert results['judge_parse_failures'] == 3
    assert 'J/f1_like' not in results
