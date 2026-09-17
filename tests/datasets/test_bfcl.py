import json

import pytest

from opencompass.datasets.bfcl import (BFCLASTEvaluator, BFCLDataset,
                                       bfcl_ast_checker, bfcl_ast_parse,
                                       is_function_calling_format_output)

SIMPLE_FUNCTION = {
    'name': 'calculate_triangle_area',
    'description': 'Calculate the area of a triangle given its base and '
    'height.',
    'parameters': {
        'type': 'dict',
        'properties': {
            'base': {
                'type': 'integer',
                'description': 'The base of the triangle.',
            },
            'height': {
                'type': 'integer',
                'description': 'The height of the triangle.',
            },
        },
        'required': ['base', 'height'],
    },
}

PARALLEL_FUNCTIONS = [
    {
        'name': 'get_weather',
        'description': 'Get the current weather of a city.',
        'parameters': {
            'type': 'dict',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'The name of the city.',
                },
            },
            'required': ['city'],
        },
    },
    {
        'name': 'convert_currency',
        'description': 'Convert an amount between currencies.',
        'parameters': {
            'type': 'dict',
            'properties': {
                'amount': {
                    'type': 'float',
                    'description': 'The amount to convert.',
                },
                'from_currency': {
                    'type': 'string',
                    'description': 'The source currency code.',
                },
                'to_currency': {
                    'type': 'string',
                    'description': 'The target currency code.',
                },
            },
            'required': ['amount', 'from_currency', 'to_currency'],
        },
    },
]


def _write_category_files(root, category, questions, answers):
    question_path = root / f'BFCL_v3_{category}.json'
    answer_dir = root / 'possible_answer'
    answer_dir.mkdir()
    answer_path = answer_dir / f'BFCL_v3_{category}.json'
    question_path.write_text('\n'.join(json.dumps(q) for q in questions),
                             encoding='utf-8')
    answer_path.write_text('\n'.join(json.dumps(a) for a in answers),
                           encoding='utf-8')
    return question_path


def test_bfcl_local_loader(tmp_path):
    questions = [{
        'id':
        'simple_0',
        'question': [[{
            'role':
            'user',
            'content':
            'Find the area of a triangle with a base of 10 units '
            'and height of 5 units.',
        }]],
        'function': [SIMPLE_FUNCTION],
    }, {
        'id':
        'simple_1',
        'question': [[{
            'role':
            'user',
            'content':
            'Find the area of a triangle with a base of 3 units '
            'and height of 7 units.',
        }]],
        'function': [SIMPLE_FUNCTION],
    }]
    answers = [
        {
            'id':
            'simple_0',
            'ground_truth': [{
                'calculate_triangle_area': {
                    'base': [10],
                    'height': [5],
                },
            }]
        },
        {
            'id':
            'simple_1',
            'ground_truth': [{
                'calculate_triangle_area': {
                    'base': [3],
                    'height': [7],
                },
            }]
        },
    ]
    _write_category_files(tmp_path, 'simple', questions, answers)

    dataset = BFCLDataset.load(path=str(tmp_path), category='simple')['test']

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample['id'] == 'simple_0'
    assert 'calculate_triangle_area' in sample['system_prompt']
    assert '"base"' in sample['system_prompt']
    assert 'Find the area of a triangle' in sample['user_prompt']
    gold = json.loads(sample['gold'])
    assert gold['category'] == 'simple'
    assert gold['functions'] == [SIMPLE_FUNCTION]
    assert gold['ground_truth'] == answers[0]['ground_truth']


def test_bfcl_local_loader_appends_system_message(tmp_path):
    questions = [{
        'id':
        'live_simple_0',
        'question': [[
            {
                'role': 'system',
                'content': 'Answer in imperial units.',
            },
            {
                'role': 'user',
                'content': 'What is the temperature in Tokyo?',
            },
        ]],
        'function': [PARALLEL_FUNCTIONS[0]],
    }]
    answers = [{
        'id': 'live_simple_0',
        'ground_truth': [{
            'get_weather': {
                'city': ['Tokyo'],
            },
        }]
    }]
    _write_category_files(tmp_path, 'live_simple', questions, answers)

    sample = BFCLDataset.load(path=str(tmp_path),
                              category='live_simple')['test'][0]

    assert sample['system_prompt'].startswith('You are an expert in '
                                              'composing functions.')
    assert sample['system_prompt'].endswith('Answer in imperial units.')
    assert sample['user_prompt'] == 'What is the temperature in Tokyo?'


def test_bfcl_rejects_unsupported_category(tmp_path):
    questions = [{
        'id': 'multi_turn_0',
        'question': [[{
            'role': 'user',
            'content': 'Hi'
        }]],
        'function': [SIMPLE_FUNCTION],
    }]
    answers = [{'id': 'multi_turn_0', 'ground_truth': [{}]}]
    _write_category_files(tmp_path, 'multi_turn_base', questions, answers)

    with pytest.raises(ValueError, match='Unsupported BFCL category'):
        BFCLDataset.load(path=str(tmp_path), category='multi_turn_base')


def test_bfcl_ast_parse_single_call():
    assert bfcl_ast_parse('func(a=1, b="x")') == [{
        'func': {
            'a': 1,
            'b': 'x',
        }
    }]


def test_bfcl_ast_parse_list_of_calls():
    assert bfcl_ast_parse('[func(a=1), func(a=2)]') == [
        {
            'func': {
                'a': 1,
            }
        },
        {
            'func': {
                'a': 2,
            }
        },
    ]


def test_bfcl_ast_parse_nested_values_and_names():
    assert bfcl_ast_parse("mod.sub.func(a=-1, b=[1, 2], c=('x', 'y'))") == [{
        'mod.sub.func': {
            'a': -1,
            'b': [1, 2],
            'c': ('x', 'y'),
        },
    }]


def test_bfcl_ast_parse_strips_wrapping_quotes():
    assert bfcl_ast_parse("'func(a=1)'") == [{
        'func': {
            'a': 1,
        }
    }]


def test_bfcl_ast_parse_rejects_invalid_output():
    with pytest.raises(Exception):
        bfcl_ast_parse('I cannot call any function.')


def test_is_function_calling_format_output():
    assert is_function_calling_format_output([{'func': {'a': 1}}])
    assert not is_function_calling_format_output([{'func': 1}])
    assert not is_function_calling_format_output([{
        'func': {
            'a': 1
        }
    }, 'not-a-dict'])
    assert not is_function_calling_format_output({'func': {'a': 1}})


def test_bfcl_ast_checker_simple_correct():
    result = bfcl_ast_checker(
        [SIMPLE_FUNCTION],
        [{
            'calculate_triangle_area': {
                'base': 10,
                'height': 5,
            }
        }],
        [{
            'calculate_triangle_area': {
                'base': [10],
                'height': [5],
            }
        }],
        'simple',
    )
    assert result['valid']


def test_bfcl_ast_checker_simple_wrong_value():
    result = bfcl_ast_checker(
        [SIMPLE_FUNCTION],
        [{
            'calculate_triangle_area': {
                'base': 10,
                'height': 6,
            }
        }],
        [{
            'calculate_triangle_area': {
                'base': [10],
                'height': [5],
            }
        }],
        'simple',
    )
    assert not result['valid']


def test_bfcl_ast_checker_simple_string_case_insensitive():
    result = bfcl_ast_checker(
        [PARALLEL_FUNCTIONS[0]],
        [{
            'get_weather': {
                'city': 'tokyo',
            }
        }],
        [{
            'get_weather': {
                'city': ['Tokyo'],
            }
        }],
        'live_simple',
    )
    assert result['valid']


def test_bfcl_ast_checker_simple_wrong_count():
    result = bfcl_ast_checker(
        [SIMPLE_FUNCTION],
        [{
            'calculate_triangle_area': {
                'base': 10,
                'height': 5,
            }
        }, {
            'calculate_triangle_area': {
                'base': 1,
                'height': 2,
            }
        }],
        [{
            'calculate_triangle_area': {
                'base': [10],
                'height': [5],
            }
        }],
        'simple',
    )
    assert not result['valid']
    assert result['error_type'] == 'simple_function_checker:wrong_count'


def test_bfcl_ast_checker_multiple():
    ground_truth = [{
        'calculate_triangle_area': {
            'base': [10],
            'height': [5],
        },
    }]
    correct = bfcl_ast_checker(
        [SIMPLE_FUNCTION],
        [{
            'calculate_triangle_area': {
                'base': 10,
                'height': 5,
            }
        }],
        ground_truth,
        'multiple',
    )
    assert correct['valid']

    wrong_count = bfcl_ast_checker(
        [SIMPLE_FUNCTION],
        [{
            'calculate_triangle_area': {
                'base': 10,
            }
        }, {
            'calculate_triangle_area': {
                'height': 5,
            }
        }],
        ground_truth,
        'multiple',
    )
    assert not wrong_count['valid']
    assert wrong_count['error_type'] == 'multiple_function_checker:wrong_count'


def test_bfcl_ast_checker_parallel_is_order_insensitive():
    ground_truth = [
        {
            'get_weather': {
                'city': ['Tokyo'],
            }
        },
        {
            'convert_currency': {
                'amount': [100.0],
                'from_currency': ['USD'],
                'to_currency': ['EUR'],
            }
        },
    ]
    reversed_output = [
        {
            'convert_currency': {
                'amount': 100,
                'from_currency': 'USD',
                'to_currency': 'EUR',
            }
        },
        {
            'get_weather': {
                'city': 'Tokyo',
            }
        },
    ]
    result = bfcl_ast_checker(PARALLEL_FUNCTIONS, reversed_output,
                              ground_truth, 'parallel')
    assert result['valid']


def test_bfcl_ast_checker_parallel_wrong_count():
    ground_truth = [
        {
            'get_weather': {
                'city': ['Tokyo'],
            }
        },
    ]
    result = bfcl_ast_checker(
        PARALLEL_FUNCTIONS,
        [{
            'get_weather': {
                'city': 'Tokyo',
            }
        }, {
            'get_weather': {
                'city': 'Osaka',
            }
        }],
        ground_truth,
        'parallel',
    )
    assert not result['valid']
    assert result['error_type'] == (
        'parallel_function_checker_no_order:wrong_count')


def test_bfcl_evaluator_score():
    evaluator = BFCLASTEvaluator()
    gold = json.dumps({
        'category':
        'simple',
        'functions': [SIMPLE_FUNCTION],
        'ground_truth': [{
            'calculate_triangle_area': {
                'base': [10],
                'height': [5],
            },
        }],
    })
    result = evaluator.score(
        predictions=[
            'calculate_triangle_area(base=10, height=5)',
            'calculate_triangle_area(base=10, height=6)',
            'The function cannot be called.',
        ],
        references=[gold, gold, gold],
    )
    assert result['accuracy'] == pytest.approx(100 / 3)
    assert result['details'][0]['correct']
    assert not result['details'][1]['correct']
    assert result['details'][1]['error_type'] == 'value_error:others'
    assert not result['details'][2]['correct']
    assert result['details'][2]['error_type'] == 'ast_decoder:decoder_failed'


def test_bfcl_evaluator_score_length_mismatch():
    evaluator = BFCLASTEvaluator()
    result = evaluator.score(predictions=['func(a=1)'], references=[])
    assert 'error' in result
