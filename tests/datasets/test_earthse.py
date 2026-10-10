import copy
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from opencompass.configs.datasets.Earth_Silver.earthse_gen import (
    SYSTEM_PROMPT,
    earthse_datasets,
)
from opencompass.datasets import (EarthSECombinedDataset, EarthSEDataset,
                                  EarthSEEvaluator, EarthSEGoldDataset,
                                  EarthSEGoldEvaluator,
                                  EarthSEGoldInferencer,
                                  REASONING_SEPARATOR,
                                  earthse_extract_content, earthse_prompt)
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.models import OpenAISDK
from opencompass.registry import ICL_EVALUATORS, ICL_INFERENCERS, LOAD_DATASET


QA_ROW = {
    'idx': 7,
    'question': 'Which letter?',
    'reasoning_chain': 'evidence',
    'answer': 'A',
    'task': 'knowledge_qa',
    'sphere': 'Atmosphere',
    'subject': 'meteorology',
    'sub_discipline': 'weather',
}

GOLD_ROW = {
    'idx': 9,
    'user_0': 'first',
    'assistant_0': 'reference first',
    'user_1': 'second',
    'assistant_1': 'reference second',
    'sphere': 'Hydrosphere',
}


def _parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_earthse_dataset_index_exposes_full_config():
    index_path = Path(__file__).resolve().parents[2] / 'dataset-index.yml'
    entries = yaml.safe_load(index_path.read_text(encoding='utf-8'))
    earthse = next(entry['earthse'] for entry in entries if 'earthse' in entry)
    config_path = 'opencompass/configs/datasets/Earth_Silver/earthse_gen.py'
    assert earthse['name'] == 'EarthSE'
    assert earthse['configpath'] == config_path
    assert earthse['configpath_llmjudge'] == config_path


def test_earthse_native_config_and_loader(tmp_path):
    source = tmp_path / 'multiple_choice.parquet'
    _parquet(source, [QA_ROW])
    cfg = copy.deepcopy(earthse_datasets[0])
    cfg.update(
        paths={'multiple_choice': str(source)},
        path=None,
        question_types=['multiple_choice'],
    )
    dataset = EarthSEDataset(**cfg)
    row = dataset.test[0]
    assert row['sample_id'] == 'earth_iron_multiple_choice:7'
    assert row['prompt'] == earthse_prompt(QA_ROW['question'],
                                           'multiple_choice')

    template = RawPromptTemplate(**{
        key: value
        for key, value in cfg['infer_cfg']['prompt_template'].items()
        if key != 'type'
    })
    assert template.generate_item(row) == [
        {
            'role': 'system',
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': earthse_prompt(QA_ROW['question'], 'multiple_choice')
        },
    ]
    assert LOAD_DATASET.get('EarthSEDataset') is EarthSEDataset
    assert LOAD_DATASET.get(
        'EarthSECombinedDataset') is EarthSECombinedDataset
    assert ICL_EVALUATORS.get('EarthSEEvaluator') is EarthSEEvaluator
    assert ICL_INFERENCERS.get(
        'EarthSEGoldInferencer') is EarthSEGoldInferencer
    evaluator_cfg = cfg['eval_cfg']['evaluator']
    assert evaluator_cfg['similarity_model_revision'] == (
        '1110a243fdf4706b3f48f1d95db1a4f5529b4d41')


def test_earthse_missing_and_malformed_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match='No artifact configured'):
        EarthSEDataset.load(
            paths={'multiple_choice': str(tmp_path / 'one.parquet')},
            dataset_name='Earth-Iron',
            question_types=['multiple_choice', 'true_false'],
        )
    with pytest.raises(FileNotFoundError, match='artifact does not exist'):
        EarthSEDataset.load(
            paths={'multiple_choice': str(tmp_path / 'missing.parquet')},
            dataset_name='Earth-Iron',
            question_types=['multiple_choice'],
        )
    malformed = tmp_path / 'bad.parquet'
    _parquet(malformed, [{'idx': 1, 'question': 'missing fields'}])
    with pytest.raises(ValueError, match='missing columns'):
        EarthSEDataset.load(
            paths={'multiple_choice': str(malformed)},
            dataset_name='Earth-Iron',
            question_types=['multiple_choice'],
        )

    missing_reasoning = tmp_path / 'missing_reasoning_chain.parquet'
    _parquet(missing_reasoning, [{
        key: value
        for key, value in QA_ROW.items() if key != 'reasoning_chain'
    }])
    with pytest.raises(ValueError, match=r"\['reasoning_chain'\]"):
        EarthSEDataset.load(
            paths={'multiple_choice': str(missing_reasoning)},
            dataset_name='Earth-Iron',
            question_types=['multiple_choice'],
        )


def test_earthse_exact_invalid_prediction_and_zero_boundaries():
    evaluator = EarthSEEvaluator(judge_model_cfg=None)
    test_set = [{
        **QA_ROW,
        'sample_id': 'earth_iron_multiple_choice:7',
        'question_type': 'multiple_choice',
    }]
    result = evaluator.score(['not a valid letter'], ['A'], test_set)
    assert result['ACC'] == 0
    assert result['multiple_choice'] == 0
    assert result['free_form (Acc.)'] == 0
    assert result['free_form (SS)'] == 0
    assert result['details'][0]['correct'] is False
    assert earthse_extract_content(
        'reasoning' + REASONING_SEPARATOR + 'answer') == 'answer'


def test_earthse_free_form_unstripped_parser_and_grouping():
    evaluator = EarthSEEvaluator(judge_model_cfg={})
    evaluator._map_judgments = lambda prompts: ['B\n']
    evaluator._similarities = lambda pairs: [0.25]
    test_set = [{
        **QA_ROW,
        'sample_id': 'earth_silver_free_form:7',
        'question_type': 'free_form',
    }]
    result = evaluator.score(['candidate'], ['reference'], test_set)
    assert 'win' not in result['details'][0]
    assert result['free_form (Acc.)'] == 0
    assert result['free_form (SS)'] == 0.25
    assert result['task/knowledge_qa'] == 0
    assert result['sphere/Atmosphere'] == 0


def test_earthse_combined_tier_aggregation(tmp_path):
    sources = []
    for dataset_name in ('Earth-Silver', 'Earth-Iron'):
        path = tmp_path / f'{dataset_name}.parquet'
        _parquet(path, [QA_ROW, {**QA_ROW, 'idx': 8}])
        sources.append(dict(
            paths={'multiple_choice': str(path)},
            dataset_name=dataset_name,
            question_types=['multiple_choice'],
        ))
    dataset = EarthSECombinedDataset.load(
        sources=sources, max_samples_per_split=1)
    evaluator = EarthSEEvaluator(judge_model_cfg=None)
    result = evaluator.score(
        ['A', 'invalid'], ['A', 'A'], dataset)
    assert len(dataset) == 2
    assert result['dataset/Earth-Silver/multiple_choice'] == 100
    assert result['dataset/Earth-Iron/multiple_choice'] == 0


def test_earth_gold_expansion_prompt_and_duplicate_key_judge(tmp_path):
    source = tmp_path / 'gold.parquet'
    _parquet(source, [GOLD_ROW])
    dataset = EarthSEGoldDataset(
        path=str(source),
        reader_cfg=dict(input_columns=['dialogue'],
                        output_column='reference'),
    )
    assert len(dataset.test) == 3
    assert [x['repetition'] for x in dataset.test] == [0, 1, 2]
    assert dataset.test[0]['dialogue'][1]['content'].endswith('first')
    assert dataset.test[0]['dialogue'][3]['content'].endswith('second')

    generated = [{
        'user_0': 'first',
        'assistant_0': f'first {i}',
        'user_1': 'second',
        'assistant_1': f'second {i}',
    } for i in range(3)]
    prompt = EarthSEGoldEvaluator.build_retention_judge_prompt(
        GOLD_ROW, generated)
    assert 'reference first' not in prompt
    assert "'assistant': 'reference second'" in prompt
    assert 'first 0' not in prompt
    assert "'assistant': 'second 0'" in prompt


def test_earth_gold_invalid_rank_omits_retention_and_ses():
    evaluator = EarthSEGoldEvaluator(judge_model_cfg={})
    evaluator._map_judgments = lambda prompts: ['rank 1']
    evaluator._diversity = lambda answers: 2.0
    predictions = [['a', 'b'], ['c', 'd'], ['e', 'f']]
    references = [GOLD_ROW] * 3
    test_set = [{
        'sample_id': 'earth_gold_train:9',
        'repetition': i
    } for i in range(3)]
    result = evaluator.score(predictions, references, test_set)
    assert result['retention_rate'] == 0
    assert result['diversity'] == 2.0
    assert result['SES'] == 0
    assert all('retention_rate' not in item for item in result['details'])
    assert all('SES' not in item for item in result['details'])


def test_earth_gold_second_turn_uses_raw_first_question_history():
    class RecordingModel:
        calls = []

        def generate_from_template(self, messages, **kwargs):
            self.calls.append((copy.deepcopy(messages), kwargs))
            return [f'answer {len(self.calls)}']

    inferencer = EarthSEGoldInferencer.__new__(EarthSEGoldInferencer)
    inferencer.model = RecordingModel()
    inferencer.batch_size = 1
    inferencer.max_out_len = 4096
    prefix = 'Please respond to the following question in 80 words or less.\n'
    entry = [[
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prefix + 'first'},
        {'role': 'assistant', 'content': ''},
        {'role': 'user', 'content': prefix + 'second'},
        {'role': 'assistant', 'content': ''},
    ]]

    assert inferencer._generate_multiround(entry, {}) == [
        ['answer 1', 'answer 2']
    ]
    second_messages, second_kwargs = inferencer.model.calls[1]
    assert second_messages[0][1]['content'] == 'first'
    assert second_messages[0][3]['content'].endswith('\nsecond')
    assert second_kwargs['temperature'] == 0


def test_earthse_judge_uses_native_model_request(monkeypatch):
    requests = []

    class Completions:

        @staticmethod
        def create(**kwargs):
            requests.append(kwargs)
            content = kwargs['messages'][1]['content'].split()[-1]
            message = SimpleNamespace(content=content, reasoning_content=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(OpenAISDK, '_create_fresh_client',
                        lambda self: client)
    evaluator = EarthSEEvaluator(judge_model_cfg=dict(
        type=OpenAISDK,
        path='judge-model',
        key='test-key',
        openai_api_base='https://example.invalid/v1',
        retry=1,
        query_per_second=0,
        temperature=0,
        timeout=900,
    ), judge_max_workers=2)

    assert evaluator._map_judgments(['judge A', 'judge B']) == ['A', 'B']
    assert evaluator.judge_model.max_workers == 2
    assert len(requests) == 2
    request = next(item for item in requests
                   if item['messages'][1]['content'] == 'judge A')
    assert request['model'] == 'judge-model'
    assert request['messages'] == [{
        'role': 'system',
        'content': SYSTEM_PROMPT,
    }, {
        'role': 'user',
        'content': 'judge A',
    }]
    assert request['temperature'] == 0
    assert request['max_tokens'] == 4096
    assert request['n'] == 1
    assert request['timeout'] == 900
