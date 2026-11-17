import json

import pytest

from opencompass.datasets.bfcl import BFCLDataset, BFCLEvaluator

QUESTIONS = [{
    'id': 'simple_0',
    'question': [[{
        'role': 'user',
        'content': 'Find the area of a triangle with base 10 and height 5.',
    }]],
    'function': [{
        'name': 'calculate_triangle_area',
        'description': 'Calculate the area of a triangle.',
        'parameters': {
            'type': 'dict',
            'properties': {
                'base': {
                    'type': 'integer'
                },
                'height': {
                    'type': 'integer'
                },
                'unit': {
                    'type': 'string'
                },
            },
            'required': ['base', 'height'],
        },
    }],
}, {
    'id': 'simple_1',
    'question': [[{
        'role': 'user',
        'content': 'What is the factorial of 5?'
    }]],
    'function': [{
        'name': 'math.factorial',
        'description': 'Factorial.',
        'parameters': {
            'type': 'dict',
            'properties': {
                'number': {
                    'type': 'integer'
                }
            },
            'required': ['number'],
        },
    }],
}]

ANSWERS = [{
    'id': 'simple_0',
    'ground_truth': [{
        'calculate_triangle_area': {
            'base': [10],
            'height': [5],
            'unit': ['units', ''],
        }
    }],
}, {
    'id': 'simple_1',
    'ground_truth': [{
        'math.factorial': {
            'number': [5]
        }
    }],
}]


def _write_jsonl(path, records):
    with open(path, 'w', encoding='utf-8') as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    return str(path)


@pytest.fixture(name='bfcl_files')
def fixture_bfcl_files(tmp_path):
    questions = _write_jsonl(tmp_path / 'questions.jsonl', QUESTIONS)
    answers = _write_jsonl(tmp_path / 'answers.jsonl', ANSWERS)
    return questions, answers


def test_loader_renders_prompt_and_keeps_reference(bfcl_files):
    questions, answers = bfcl_files
    dataset = BFCLDataset.load(path=questions, answer_path=answers)

    assert len(dataset) == 2
    first = dataset[0]
    assert first['id'] == 'simple_0'
    assert 'calculate_triangle_area' in first['prompt']
    assert 'base of a triangle' in first['prompt'].lower().replace(
        'area of a triangle', 'base of a triangle')
    assert json.loads(first['gts']) == ANSWERS[0]['ground_truth']


def test_loader_skips_samples_without_reference(tmp_path):
    questions = _write_jsonl(tmp_path / 'q.jsonl', QUESTIONS)
    answers = _write_jsonl(tmp_path / 'a.jsonl', ANSWERS[:1])

    dataset = BFCLDataset.load(path=questions, answer_path=answers)

    assert len(dataset) == 1
    assert dataset[0]['id'] == 'simple_0'


@pytest.mark.parametrize(
    'prediction,expected',
    [
        ('{"name": "calculate_triangle_area", '
         '"arguments": {"base": 10, "height": 5}}', True),
        ('{"name": "calculate_triangle_area", '
         '"arguments": {"base": 10, "height": 5, "unit": "units"}}', True),
        ('```json\n{"name": "calculate_triangle_area", '
         '"arguments": {"base": 10, "height": 5}}\n```', True),
        ('Calling now: {"name": "calculate_triangle_area", '
         '"arguments": {"base": 10, "height": 5}} done', True),
        ('{"name": "calculate_triangle_area", "arguments": {"base": 10}}',
         False),
        ('{"name": "wrong_name", "arguments": {"base": 10, "height": 5}}',
         False),
        ('{"name": "calculate_triangle_area", '
         '"arguments": {"base": 11, "height": 5}}', False),
    ])
def test_evaluator_scores_one_sample(prediction, expected):
    evaluator = BFCLEvaluator()
    reference = json.dumps(ANSWERS[0]['ground_truth'])

    result = evaluator.score([prediction], [reference])

    assert result['details'][0]['correct'] is expected
    assert result['accuracy'] == (100.0 if expected else 0.0)


def test_evaluator_treats_numbers_and_strings_alike():
    evaluator = BFCLEvaluator()
    reference = json.dumps([{'math.factorial': {'number': [5]}}])

    result = evaluator.score(
        [
            '{"name": "math.factorial", "arguments": {"number": 5.0}}',
            '{"name": "math.factorial", "arguments": {"number": "5"}}',
            '{"name": "math.factorial", "arguments": {"number": 6}}',
        ],
        [reference] * 3,
    )

    assert [detail['correct'] for detail in result['details']
            ] == [True, True, False]
    assert result['accuracy'] == pytest.approx(100 * 2 / 3)


def test_evaluator_counts_unparseable_replies():
    evaluator = BFCLEvaluator()
    reference = json.dumps(ANSWERS[1]['ground_truth'])

    result = evaluator.score(['I cannot help with that.'], [reference])

    assert result['accuracy'] == 0.0
    assert result['parse_failure_rate'] == 100.0


def test_evaluator_rejects_length_mismatch():
    evaluator = BFCLEvaluator()

    result = evaluator.score(['{}'], ['[]', '[]'])

    assert 'error' in result
