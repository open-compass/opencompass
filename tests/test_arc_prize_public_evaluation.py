import json
import runpy
from pathlib import Path

from datasets import Dataset

from opencompass.datasets.arc_prize_public_evaluation import (
    ARCPrizeDataset,
    ARCPrizeEvaluator,
    ARCPrizeGenInferencer,
    SECOND_PASS_EXTRACTION_PROMPT,
    _parse_second_pass_response,
    build_owner_prompt,
    extract_solution,
)
from opencompass.utils.prompt import PromptList


def test_owner_prompt_matches_baseline_shape():
    prompt = build_owner_prompt(
        [{
            'input': [[1]],
            'output': [[2]]
        }],
        [[3]],
    )

    assert '--Example 0-- \n\n INPUT: \n\n[[1]]' in prompt
    assert 'OUTPUT: \n\n[[2]]' in prompt
    assert '--Test Input--\n\n[[3]]' in prompt
    assert prompt.endswith('Your response:')
    assert "'input'" not in prompt


def test_loader_expands_every_test_pair_and_keeps_task_identity(tmp_path):
    task = {
        'train': [{
            'input': [[1]],
            'output': [[2]]
        }],
        'test': [
            {
                'input': [[3]],
                'output': [[4]]
            },
            {
                'input': [[5]],
                'output': [[6]]
            },
        ],
    }
    (tmp_path / '00576224.json').write_text(json.dumps(task))

    dataset = ARCPrizeDataset.load(
        str(tmp_path),
        'arc_agi_1',
        protocol='owner',
    )

    assert len(dataset) == 2
    assert dataset[0]['task_id'] == dataset[1]['task_id'] == '00576224'
    assert dataset[0]['pair_index'] == 0
    assert dataset[1]['pair_index'] == 1
    assert dataset[1]['output_test_data']['output'] == [[6]]


def test_default_loader_preserves_legacy_first_pair_and_schema(tmp_path):
    task = {
        'train': [{
            'input': [[1]],
            'output': [[2]]
        }],
        'test': [
            {
                'input': [[3]],
                'output': [[4]]
            },
            {
                'input': [[5]],
                'output': [[6]]
            },
        ],
    }
    (tmp_path / '00576224.json').write_text(json.dumps(task))

    dataset = ARCPrizeDataset.load(str(tmp_path), 'arc_agi_1')

    assert len(dataset) == 1
    assert dataset[0]['input_test_data'] == [[3]]
    assert dataset[0]['output_test_data'] == [[4]]
    assert 'prompt' not in dataset.column_names
    assert 'task_id' not in dataset.column_names


def test_extract_solution_matches_owner_structure_validation():
    assert extract_solution('```json\n[[1, 2], [3, 4]]\n```') == [[1, 2],
                                                                  [3, 4]]
    assert extract_solution('Training grid: [[1]]. Final answer: [[2]].') == [[
        2
    ]]
    assert extract_solution(
        r'Final answer: \boxed{[[3]]}. Trailing grid: [[4]].') == [[3]]
    assert extract_solution('answer: [[1], [2, 3]]') == [[1], [2, 3]]
    assert extract_solution('answer: [[10]]') == [[10]]
    assert extract_solution(r'answer: \boxed{[]}') == []
    assert extract_solution('no grid') is None


def test_second_pass_prompt_and_parser_match_owner_openai_adapter():
    prompt = SECOND_PASS_EXTRACTION_PROMPT.format(
        response='analysis without a locally parseable grid')
    assert 'Extract only the JSON array of arrays' in prompt
    assert 'analysis without a locally parseable grid' in prompt
    assert _parse_second_pass_response('```json\n[[1, 2], [3, 4]]\n```') == [[
        1, 2
    ], [3, 4]]
    assert _parse_second_pass_response('{"response": [[5, 6], [7, 8]]}') == [[
        5, 6
    ], [7, 8]]
    assert _parse_second_pass_response('not a grid') is None


def test_inferencer_reuses_model_for_unparseable_responses_only():
    inferencer = ARCPrizeGenInferencer.__new__(ARCPrizeGenInferencer)
    inferencer.enable_second_pass = True
    inferencer.second_pass_max_out_len = 4096
    calls = []

    def generate(templates, max_out_len, **kwargs):
        calls.append((templates, max_out_len, kwargs))
        if len(calls) == 1:
            return ['[[1]]', 'analysis without a grid']
        return ['{"response": [[2]]}']

    predictions = inferencer._generate_with_second_pass(
        generate,
        ['primary prompt 1', 'primary prompt 2'],
        8192,
        temperature=0.0,
    )

    assert predictions == ['[[1]]', '[[2]]']
    assert len(calls) == 2
    assert calls[0][1] == 8192
    assert calls[1][1] == 4096
    assert calls[1][2] == {'temperature': 0.0}
    extraction_prompts = calls[1][0]
    assert len(extraction_prompts) == 1
    assert isinstance(extraction_prompts[0], PromptList)
    assert extraction_prompts[0][0]['role'] == 'HUMAN'
    assert 'analysis without a grid' in extraction_prompts[0][0]['prompt']


def test_inferencer_can_disable_second_pass():
    inferencer = ARCPrizeGenInferencer.__new__(ARCPrizeGenInferencer)
    inferencer.enable_second_pass = False
    inferencer.second_pass_max_out_len = 4096
    calls = []

    def generate(templates, max_out_len, **kwargs):
        calls.append((templates, max_out_len, kwargs))
        return ['analysis without a grid']

    predictions = inferencer._generate_with_second_pass(
        generate,
        ['primary prompt'],
        8192,
    )

    assert predictions == ['analysis without a grid']
    assert len(calls) == 1


def test_inferencer_preserves_prediction_metadata_after_second_pass():
    inferencer = ARCPrizeGenInferencer.__new__(ARCPrizeGenInferencer)
    inferencer.enable_second_pass = True
    inferencer.second_pass_max_out_len = 4096
    calls = 0

    def generate(templates, max_out_len, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{'prediction': 'analysis only', 'tokens': 12}]
        return [{'prediction': '[[3]]', 'tokens': 4}]

    predictions = inferencer._generate_with_second_pass(
        generate,
        ['primary prompt'],
        8192,
    )

    assert predictions == [{'prediction': '[[3]]', 'tokens': 12}]


def test_evaluator_aggregates_pairs_within_each_task():
    references = [
        {
            'task_id': 'a',
            'pair_index': 0,
            'num_pairs': 2,
            'output': [[1]]
        },
        {
            'task_id': 'a',
            'pair_index': 1,
            'num_pairs': 2,
            'output': [[2]]
        },
        {
            'task_id': 'b',
            'pair_index': 0,
            'num_pairs': 1,
            'output': [[3]]
        },
    ]
    result = ARCPrizeEvaluator(protocol='owner').score(
        predictions=['[[1]]', '[[9]]', '[[3]]'],
        references=references,
    )

    assert result['accuracy'] == 0.75
    assert result['pair_accuracy'] == 2 / 3
    assert result['task_count'] == 2
    assert result['pair_count'] == 3
    assert result['parse_success_rate'] == 1.0


def test_owner_evaluator_keeps_replica_average_without_attempt_any():
    reference = {
        'task_id': 'a',
        'pair_index': 0,
        'num_pairs': 1,
        'output': [[1]],
    }
    result = ARCPrizeEvaluator(protocol='owner').evaluate(
        k=2,
        n=2,
        original_dataset=Dataset.from_list([
            {
                'idx': 0,
                'subdivision': 'arc'
            },
            {
                'idx': 0,
                'subdivision': 'arc'
            },
        ]),
        predictions=['[[1]]', '[[9]]'],
        references=[reference, reference],
    )

    assert result['accuracy (2 runs average)'] == 0.5
    assert 'G-Pass@2_1.0' in result
    assert 'attempts_per_pair' not in result


def test_default_evaluator_preserves_legacy_first_grid_parser():
    result = ARCPrizeEvaluator().score(
        predictions=['Training grid: [[1]]. Final answer: [[2]].'],
        references=[[[1]]],
    )

    assert result['accuracy'] == 1.0
    assert result['details'][0]['generated_solution'] == [[1]]
    assert set(result) == {'accuracy', 'details'}


def test_evaluator_aggregates_replicas_as_owner_attempts():
    base_references = [
        {
            'task_id': 'a',
            'pair_index': 0,
            'num_pairs': 2,
            'output': [[1]]
        },
        {
            'task_id': 'a',
            'pair_index': 1,
            'num_pairs': 2,
            'output': [[2]]
        },
        {
            'task_id': 'b',
            'pair_index': 0,
            'num_pairs': 1,
            'output': [[3]]
        },
    ]
    references = base_references * 2
    original_dataset = Dataset.from_list([{
        'idx': index,
        'subdivision': 'arc'
    } for index in range(len(references))])

    result = ARCPrizeEvaluator(
        protocol='owner',
        attempt_aggregation='any',
        attempts_per_pair=2,
    ).evaluate(
        k=2,
        n=2,
        original_dataset=original_dataset,
        predictions=[
            '[[9]]',
            '[[2]]',
            '[[9]]',
            '[[1]]',
            '[[9]]',
            '[[9]]',
        ],
        references=references,
    )

    assert result['accuracy'] == 0.5
    assert result['pair_accuracy'] == 2 / 3
    assert result['task_count'] == 2
    assert result['pair_count'] == 3
    assert result['attempt_count'] == 6
    assert result['attempts_per_pair'] == 2
    assert result['details'][0]['attempt_correct'] == [0, 1]
    assert result['details'][0]['correct'] == 1
    assert result['details'][1]['attempt_correct'] == [1, 0]
    assert result['details'][2]['correct'] == 0
    assert 'accuracy (2 runs average)' not in result
    assert 'G-Pass@2_1.0' not in result


def test_arc_agi_2_config_uses_owner_protocol():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root / 'opencompass/configs/datasets/'
        'ARC_Prize_Public_Evaluation/arc_agi_2_public_evaluation_gen.py')
    config = runpy.run_path(str(config_path))
    dataset_cfg = config['arc_agi_2_public_evaluation_datasets'][0]
    inferencer_cfg = dataset_cfg['infer_cfg']['inferencer']
    evaluator_cfg = dataset_cfg['eval_cfg']['evaluator']

    assert dataset_cfg['reader_cfg']['input_columns'] == ['prompt']
    assert inferencer_cfg['type'] is ARCPrizeGenInferencer
    assert inferencer_cfg['enable_second_pass'] is True
    assert inferencer_cfg['second_pass_max_out_len'] == 4096
    assert 'max_out_len' not in inferencer_cfg
    assert dataset_cfg['n'] == dataset_cfg['k'] == 2
    assert dataset_cfg['protocol'] == 'owner'
    assert evaluator_cfg['protocol'] == 'owner'
    assert evaluator_cfg['attempt_aggregation'] == 'any'
    assert evaluator_cfg['attempts_per_pair'] == 2


def test_arc_agi_1_config_uses_owner_protocol():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root /
                   'opencompass/configs/datasets/ARC_Prize_Public_Evaluation/'
                   'arc_prize_public_evaluation_gen_872059.py')
    config = runpy.run_path(str(config_path))
    dataset_cfg = config['arc_prize_public_evaluation_datasets'][0]
    inferencer_cfg = dataset_cfg['infer_cfg']['inferencer']
    evaluator_cfg = dataset_cfg['eval_cfg']['evaluator']

    assert dataset_cfg['reader_cfg']['input_columns'] == ['prompt']
    assert inferencer_cfg['type'] is ARCPrizeGenInferencer
    assert inferencer_cfg['enable_second_pass'] is True
    assert inferencer_cfg['second_pass_max_out_len'] == 4096
    assert 'max_out_len' not in inferencer_cfg
    assert dataset_cfg['protocol'] == 'owner'
    assert evaluator_cfg['protocol'] == 'owner'


def test_legacy_config_does_not_opt_into_owner_protocol():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / 'opencompass/configs/datasets/'
                   'ARC_Prize_Public_Evaluation/'
                   'arc_prize_public_evaluation_gen_fedd04.py')
    config = runpy.run_path(str(config_path))
    dataset_cfg = config['arc_prize_public_evaluation_datasets'][0]
    evaluator_cfg = dataset_cfg['eval_cfg']['evaluator']

    assert 'protocol' not in dataset_cfg
    assert 'protocol' not in evaluator_cfg
