"""BFCL (Berkeley Function Calling Leaderboard) single-turn evaluation.

This module adds support for the single-turn AST-evaluated categories of the
Berkeley Function Calling Leaderboard (BFCL) v3 dataset:

- ``simple`` / ``multiple`` / ``parallel`` / ``parallel_multiple``
- ``live_simple`` / ``live_multiple`` / ``live_parallel`` /
  ``live_parallel_multiple``

The decoding (``bfcl_ast_parse``) and checking (``bfcl_ast_checker``) logic is
a Python-only port of the official BFCL evaluator from
https://github.com/ShishirPatil/gorilla
(Apache License 2.0), adapted to the OpenCompass dataset/evaluator
interfaces. Unlike the official evaluator, function names are compared as they
appear in the prompt, because OpenCompass feeds function documentation to the
model as plain text rather than through provider-specific tool APIs (which is
where the official evaluator's ``_``/``.`` name conversion is needed).

Dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard on Hugging Face
Paper: https://arxiv.org/abs/2402.04653
"""

import ast
import json
import os
import re
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

DEFAULT_REPO_ID = 'gorilla-llm/Berkeley-Function-Calling-Leaderboard'

# v3 single-turn categories whose possible answers are published on the Hub.
SUPPORTED_CATEGORIES = (
    'simple',
    'multiple',
    'parallel',
    'parallel_multiple',
    'live_simple',
    'live_multiple',
    'live_parallel',
    'live_parallel_multiple',
)

BFCL_SYSTEM_PROMPT = (
    'You are an expert in composing functions. You are given a question '
    'and a set of possible functions. Based on the question, you will '
    'need to make one or more function/tool calls to achieve the '
    'purpose.\n'
    'If none of the functions can be used, point it out. If the given '
    'question lacks the parameters required by the function, also point '
    'it out.\n'
    'You should only return the function calls in your response.\n'
    '\n'
    'If you decide to invoke any of the function(s), you MUST put it in '
    'the format of [func_name1(params_name1=params_value1, '
    'params_name2=params_value2...), func_name2(params)]\n'
    'You SHOULD NOT include any other text in the response.\n'
    '\n'
    'At each turn, you should try your best to complete the tasks '
    'requested by the user within the current turn. Continue to output '
    'functions to call until you have fulfilled the user\'s request to '
    'the best of your ability. Once you have no more functions to call, '
    'the system will consider the current turn complete and proceed to '
    'the next turn or task.\n'
    '\n'
    'Here is a list of functions in JSON format that you can invoke.\n'
    '{functions}\n')


def _resolve_category_files(path: str, category: str) -> tuple:
    """Resolve the question file and the possible-answer file for a category.

    ``path`` can be a local directory containing ``BFCL_v3_<category>.json``
    and ``possible_answer/BFCL_v3_<category>.json``, a local question file
    (with the possible-answer file looked up next to it), or a Hugging Face
    dataset repo id (defaults to the official BFCL dataset).
    """
    data_name = f'BFCL_v3_{category}.json'
    answer_name = os.path.join('possible_answer', data_name)

    if os.path.isdir(path):
        data_path = os.path.join(path, data_name)
        answer_path = os.path.join(path, answer_name)
        if os.path.isfile(data_path) and os.path.isfile(answer_path):
            return data_path, answer_path
        # Also allow a flat layout with both files in the same directory.
        flat_answer_path = os.path.join(path, f'possible_answer_{data_name}')
        if os.path.isfile(data_path) and os.path.isfile(flat_answer_path):
            return data_path, flat_answer_path
        raise FileNotFoundError(
            f'{path} must contain {data_name} and possible_answer/'
            f'{data_name} (or possible_answer_{data_name}).')

    if os.path.isfile(path):
        answer_path = os.path.join(os.path.dirname(path), answer_name)
        if not os.path.isfile(answer_path):
            raise FileNotFoundError(
                f'Expected the possible-answer file at {answer_path} when '
                f'using {path} as the question file.')
        return path, answer_path

    from huggingface_hub import hf_hub_download

    data_path = hf_hub_download(repo_id=path,
                                filename=data_name,
                                repo_type='dataset')
    answer_path = hf_hub_download(repo_id=path,
                                  filename=answer_name.replace(os.sep, '/'),
                                  repo_type='dataset')
    return data_path, answer_path


def _read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_prompt(functions: List[Dict], question: List[Dict]) -> Dict:
    """Build (system_prompt, user_prompt) from one BFCL entry.

    ``question`` is a single turn, i.e. a list of ``{'role', 'content'}``
    messages. Any system-role message is appended to the BFCL system prompt.
    """
    system_parts = [BFCL_SYSTEM_PROMPT.format(functions=json.dumps(functions))]
    user_parts = []
    for message in question:
        if message.get('role') == 'system':
            system_parts.append(message['content'])
        else:
            user_parts.append(message['content'])
    return '\n\n'.join(system_parts), '\n'.join(user_parts)


@LOAD_DATASET.register_module()
class BFCLDataset(BaseDataset):

    @staticmethod
    def load(path: str = DEFAULT_REPO_ID,
             category: str = 'live_simple') -> DatasetDict:
        if category not in SUPPORTED_CATEGORIES:
            raise ValueError(
                f'Unsupported BFCL category {category!r}. Choose one of '
                f'{SUPPORTED_CATEGORIES}. Multi-turn, Java/JavaScript and '
                'relevance categories are not supported yet.')

        data_path, answer_path = _resolve_category_files(path, category)
        questions = _read_jsonl(data_path)
        answers = _read_jsonl(answer_path)
        if len(questions) != len(answers):
            raise ValueError(
                f'Question file ({len(questions)} rows) and possible-answer '
                f'file ({len(answers)} rows) have different lengths.')

        raw_data = []
        for question_entry, answer_entry in zip(questions, answers):
            # NOTE: the official BFCL evaluator aligns questions and possible
            # answers by order (they share the generation order) rather than
            # by id; the published v3 live_multiple split even contains one
            # mismatched trailing id, so an id check would reject it.
            system_prompt, user_prompt = _build_prompt(
                question_entry['function'], question_entry['question'][0])
            gold = json.dumps({
                'category': category,
                'functions': question_entry['function'],
                'ground_truth': answer_entry['ground_truth'],
            })
            raw_data.append({
                'id': question_entry['id'],
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'gold': gold,
            })

        dataset = Dataset.from_list(raw_data)
        return DatasetDict({'test': dataset, 'train': dataset})


# Output decoding (ported from the official BFCL evaluator)


def _resolve_ast_by_type(value: ast.expr) -> Any:
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            return '...'
        return value.value
    elif isinstance(value, ast.UnaryOp):
        return -value.operand.value
    elif isinstance(value, ast.List):
        return [_resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        return {
            _resolve_ast_by_type(k): _resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(value, ast.Tuple):
        return tuple(_resolve_ast_by_type(v) for v in value.elts)
    else:
        raise ValueError(f'Unsupported AST node: {type(value).__name__}')


def _resolve_ast_call(elem: ast.Call) -> Dict[str, Dict]:
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = '.'.join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        args_dict[arg.arg] = _resolve_ast_by_type(arg.value)
    return {func_name: args_dict}


def bfcl_ast_parse(input_str: str) -> List[Dict]:
    """Decode a Python-style function-call output into the standard format.

    The output may contain one call ``func(a=1)`` or a list of calls
    ``[func(a=1), func(b=2)]``. Raises on invalid syntax.
    """
    # We only want to remove wrapping quotes that could have been added by
    # the model.
    cleaned_input = input_str.strip().strip("'")
    parsed = ast.parse(cleaned_input, mode='eval')
    extracted = []
    if isinstance(parsed.body, ast.Call):
        extracted.append(_resolve_ast_call(parsed.body))
    else:
        for elem in parsed.body.elts:
            if not isinstance(elem, ast.Call):
                raise ValueError(
                    f'Unsupported AST element: {type(elem).__name__}.')
            extracted.append(_resolve_ast_call(elem))
    return extracted


def is_function_calling_format_output(decoded_output) -> bool:
    """Check the decoded output is a list of single-key dicts of dicts."""
    if type(decoded_output) is not list:
        return False
    for item in decoded_output:
        if type(item) is not dict:
            return False
        if len(item) != 1:
            return False
        if type(list(item.values())[0]) is not dict:
            return False
    return True


# AST checking (ported from the official BFCL evaluator)

PYTHON_TYPE_MAPPING = {
    'string': str,
    'integer': int,
    'float': float,
    'boolean': bool,
    'array': list,
    'tuple': list,
    'dict': dict,
    'any': str,
}

# Types whose values need to be recursively checked
PYTHON_NESTED_TYPE_CHECK_LIST = ['array', 'tuple']


def _find_description(func_descriptions, name: str) -> Optional[Dict]:
    if type(func_descriptions) is list:
        for func_description in func_descriptions:
            if func_description['name'] == name:
                return func_description
        return None
    else:
        # It is a dict; there is only one function.
        return func_descriptions


def _get_possible_answer_type(possible_answer: list):
    for answer in possible_answer:
        if answer != '':  # Optional parameter
            return type(answer)
    return None


def _type_checker(param: str, value, possible_answer: list,
                  expected_type_description: str, expected_type_converted,
                  nested_type_converted):
    """Check a parameter value against the possible answers.

    NOTE: This type checker only supports nested type checking for one level
    deep, mirroring the official BFCL evaluator.
    """
    result = {
        'valid': True,
        'error': [],
        'is_variable': False,
        'error_type': 'type_error:simple',
    }

    is_variable = False
    # Check for the case where a variable is used instead of an actual value.
    # Use the type in possible_answer as the expected type.
    possible_answer_type = _get_possible_answer_type(possible_answer)
    # If possible_answer only contains optional parameters, we can't
    # determine the type.
    if possible_answer_type is not None:
        if possible_answer_type != expected_type_converted:
            is_variable = True

    # Value is the same type as in function description.
    if type(value) is expected_type_converted:
        if nested_type_converted is None:
            result['is_variable'] = is_variable
            return result
        else:
            for possible_answer_item in possible_answer:
                flag = True
                if type(possible_answer_item) is list:
                    for value_item in value:
                        checker_result = _type_checker(
                            param,
                            value_item,
                            possible_answer_item,
                            str(nested_type_converted),
                            nested_type_converted,
                            None,
                        )
                        if not checker_result['valid']:
                            flag = False
                            break
                if flag:
                    return {
                        'valid': True,
                        'error': [],
                        'is_variable': is_variable
                    }

            result['valid'] = False
            result['error'] = [
                f'Nested type checking failed for parameter {repr(param)}. '
                f'Expected outer type {expected_type_description} with inner '
                f'type {str(nested_type_converted)}. Parameter value: '
                f'{repr(value)}.'
            ]
            result['error_type'] = 'type_error:nested'

    # Value is not as expected; check for the case where a variable is used
    # instead of an actual value.
    possible_answer_type = _get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if type(value) is possible_answer_type:
            result['is_variable'] = True
            return result

    result['valid'] = False
    result['error'].append(
        f'Incorrect type for parameter {repr(param)}. Expected type '
        f'{expected_type_description}, got {type(value).__name__}. Parameter '
        f'value: {repr(value)}.')
    result['error_type'] = 'type_error:simple'
    return result


def _standardize_string(input_string: str) -> str:
    """Remove spaces/some punctuation and lowercase, for string comparison."""
    regex_string = r'[ \,\.\/\-\_\*\^]'
    return re.sub(regex_string, '', input_string).lower().replace("'", '"')


def _string_checker(param: str, model_output: str, possible_answer: list):
    standardize_possible_answer = []
    standardize_model_output = _standardize_string(model_output)
    for i in range(len(possible_answer)):
        if type(possible_answer[i]) is str:
            standardize_possible_answer.append(
                _standardize_string(possible_answer[i]))

    if standardize_model_output not in standardize_possible_answer:
        return {
            'valid':
            False,
            'error': [
                f'Invalid value for parameter {repr(param)}: '
                f'{repr(model_output)}. Expected one of {possible_answer}. '
                'Case insensitive.'
            ],
            'error_type':
            'value_error:string',
        }

    return {'valid': True, 'error': []}


def _list_checker(param: str, model_output: list, possible_answer: list):
    standardize_model_output = list(model_output)

    for i in range(len(standardize_model_output)):
        if type(standardize_model_output[i]) is str:
            standardize_model_output[i] = _standardize_string(model_output[i])

    standardize_possible_answer = []
    for i in range(len(possible_answer)):
        standardize_possible_answer.append([])
        for j in range(len(possible_answer[i])):
            if type(possible_answer[i][j]) is str:
                standardize_possible_answer[i].append(
                    _standardize_string(possible_answer[i][j]))
            else:
                standardize_possible_answer[i].append(possible_answer[i][j])

    if standardize_model_output not in standardize_possible_answer:
        return {
            'valid':
            False,
            'error': [
                f'Invalid value for parameter {repr(param)}: '
                f'{repr(model_output)}. Expected one of {possible_answer}.'
            ],
            'error_type':
            'value_error:list/tuple',
        }

    return {'valid': True, 'error': []}


def _dict_checker(param: str, model_output: dict, possible_answers: list):
    # Works for simple dictionaries, mirroring the official evaluator.
    result = {
        'valid': False,
        'error': [],
        'error_type': 'dict_checker:unclear'
    }
    for i in range(len(possible_answers)):

        if possible_answers[i] == '':
            continue

        result = {
            'valid': False,
            'error': [],
            'error_type': 'dict_checker:unclear'
        }

        flag = True

        possible_answer = possible_answers[i]

        for key, value in model_output.items():
            if key not in possible_answer:
                result['valid'] = False
                result['error'].append(
                    f"Unexpected dict key parameter: '{key}'.")
                result['error_type'] = 'value_error:dict_key'
                flag = False
                break

            standardize_value = value
            if type(value) is str:
                standardize_value = _standardize_string(value)

            standardize_possible_answer = []
            for i in range(len(possible_answer[key])):
                if type(possible_answer[key][i]) is str:
                    standardize_possible_answer.append(
                        _standardize_string(possible_answer[key][i]))
                else:
                    standardize_possible_answer.append(possible_answer[key][i])

            if standardize_value not in standardize_possible_answer:
                result['valid'] = False
                result['error'].append(
                    f'Invalid value for parameter {repr(key)}: {repr(value)}.'
                    f' Expected one of {standardize_possible_answer}.')
                result['error_type'] = 'value_error:dict_value'
                flag = False
                break

        for key, value in possible_answer.items():
            if key not in model_output and '' not in value:
                result['valid'] = False
                result['error'].append(f"Missing dict key parameter: '{key}'.")
                result['error_type'] = 'value_error:dict_key'
                flag = False
                break

        if flag:
            return {'valid': True, 'error': []}

    return result


def _list_dict_checker(param: str, model_output: list, possible_answers: list):
    # Each dictionary in the list must match, in order.
    result = {
        'valid': False,
        'error': [],
        'error_type': 'list_dict_checker:unclear',
    }

    for answer_index in range(len(possible_answers)):
        flag = True

        if len(model_output) != len(possible_answers[answer_index]):
            result['valid'] = False
            result['error'] = ['Wrong number of dictionaries in the list.']
            result['error_type'] = 'value_error:list_dict_count'
            flag = False
            continue

        for dict_index in range(len(model_output)):
            result = _dict_checker(
                param,
                model_output[dict_index],
                [possible_answers[answer_index][dict_index]],
            )
            if not result['valid']:
                flag = False
                break
        if flag:
            return {'valid': True, 'error': []}

    return result


def _simple_function_checker(func_description: dict, model_output: dict,
                             possible_answer: dict):
    possible_answer = list(possible_answer.values())[0]
    # Extract function name and parameters details
    func_name = func_description['name']
    param_details = func_description['parameters']['properties']
    required_params = func_description['parameters']['required']

    result = {
        'valid': True,
        'error': [],
        'error_type': 'simple_function_checker:unclear',
    }

    if func_name not in model_output:
        result['valid'] = False
        result['error'].append(
            f'Function name {repr(func_name)} missing from model output.')
        result['error_type'] = 'simple_function_checker:wrong_func_name'
        return result

    model_params = model_output[func_name]

    for param in required_params:
        if param not in model_params:
            result['valid'] = False
            result['error'].append(
                f'Missing required parameter: {repr(param)}.')
            result['error_type'] = 'simple_function_checker:missing_required'
            return result

    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            result['valid'] = False
            result['error'].append(f'Unexpected parameter: {repr(param)}.')
            result['error_type'] = 'simple_function_checker:unexpected_param'
            return result

        full_param_details = param_details[param]
        expected_type_description = full_param_details['type']
        is_variable = False
        nested_type_converted = None

        expected_type_converted = PYTHON_TYPE_MAPPING[
            expected_type_description]
        if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
            nested_type = param_details[param]['items']['type']
            nested_type_converted = PYTHON_TYPE_MAPPING[nested_type]

        # We convert all tuple values to lists when the expected type is
        # tuple; any tuple in the possible answer becomes a list after JSON
        # round-tripping.
        if expected_type_description == 'tuple' and type(value) is tuple:
            value = list(value)

        # Allow Python auto conversion from int to float
        if (expected_type_description == 'float' and type(value) is int):
            value = float(value)

        type_check_result = _type_checker(
            param,
            value,
            possible_answer[param],
            expected_type_description,
            expected_type_converted,
            nested_type_converted,
        )
        is_variable = type_check_result['is_variable']
        if not type_check_result['valid']:
            return type_check_result

        # It doesn't make sense to specially handle dictionaries and lists of
        # dictionaries if the value is a variable.
        if not is_variable:
            if expected_type_converted == dict:
                result = _dict_checker(param, value, possible_answer[param])
                if not result['valid']:
                    return result
                continue

            elif (expected_type_converted == list
                  and nested_type_converted == dict):
                result = _list_dict_checker(param, value,
                                            possible_answer[param])
                if not result['valid']:
                    return result
                continue

            elif expected_type_converted == str:
                # Case-insensitive comparison for strings
                result = _string_checker(param, value, possible_answer[param])
                if not result['valid']:
                    return result
                continue

            elif expected_type_converted == list:
                result = _list_checker(param, value, possible_answer[param])
                if not result['valid']:
                    return result
                continue

        # Check if the value is within the possible answers
        if value not in possible_answer[param]:
            result['valid'] = False
            result['error'].append(
                f'Invalid value for parameter {repr(param)}: {repr(value)}. '
                f'Expected one of {possible_answer[param]}.')
            result['error_type'] = 'value_error:others'
            return result

    # Check for optional parameters not provided but allowed
    for param in possible_answer:
        if param not in model_params and '' not in possible_answer[param]:
            result['valid'] = False
            result['error'].append(
                f'Optional parameter {repr(param)} not provided and not '
                'marked as optional.')
            result['error_type'] = 'simple_function_checker:missing_optional'
            return result

    return result


def _parallel_function_checker_no_order(func_descriptions: list,
                                        model_output: list,
                                        possible_answers: list):
    if len(model_output) != len(possible_answers):
        return {
            'valid': False,
            'error': ['Wrong number of functions.'],
            'error_type': 'parallel_function_checker_no_order:wrong_count',
        }

    matched_indices = []

    # Go through the possible answers one by one, and eliminate the model
    # output that matches the possible answer.
    for i in range(len(possible_answers)):
        # possible_answers[i] is a dictionary with only one key
        func_name_expected = list(possible_answers[i].keys())[0]
        func_description = _find_description(func_descriptions,
                                             func_name_expected)

        all_errors = []

        for index in range(len(model_output)):
            if index in matched_indices:
                continue

            result = _simple_function_checker(
                func_description,
                model_output[index],
                possible_answers[i],
            )

            if result['valid']:
                matched_indices.append(index)
                break
            else:
                all_errors.append({
                    f'Model Result Index {index}': {
                        'sub_error': result['error'],
                        'sub_error_type': result['error_type'],
                        'model_output_item': model_output[index],
                        'possible_answer_item': possible_answers[i],
                    }
                })

        if not result['valid']:
            considered_indices = [
                i for i in range(len(model_output)) if i not in matched_indices
            ]
            all_errors.insert(
                0,
                f'Could not find a matching function among index '
                f'{considered_indices} of model output for index {i} of '
                'possible answers.',
            )
            return {
                'valid':
                False,
                'error':
                all_errors,
                'error_type':
                'parallel_function_checker_no_order:cannot_find_match',
            }

    return {'valid': True, 'error': []}


def _multiple_function_checker(func_descriptions: list, model_output: list,
                               possible_answers: list):
    if len(model_output) != len(possible_answers):
        return {
            'valid': False,
            'error': ['Wrong number of functions.'],
            'error_type': 'multiple_function_checker:wrong_count',
        }

    # possible_answers is a list of only one dictionary with only one key
    func_name_expected = list(possible_answers[0].keys())[0]
    func_description = _find_description(func_descriptions, func_name_expected)
    return _simple_function_checker(
        func_description,
        model_output[0],
        possible_answers[0],
    )


def bfcl_ast_checker(func_description, model_output, possible_answer,
                     test_category: str) -> Dict:
    """Run the official BFCL AST check for one entry (Python categories)."""
    if 'parallel' in test_category:
        return _parallel_function_checker_no_order(func_description,
                                                   model_output,
                                                   possible_answer)

    elif 'multiple' in test_category:
        return _multiple_function_checker(func_description, model_output,
                                          possible_answer)

    else:
        if len(model_output) != 1:
            return {
                'valid': False,
                'error': ['Wrong number of functions.'],
                'error_type': 'simple_function_checker:wrong_count',
            }

        return _simple_function_checker(func_description[0], model_output[0],
                                        possible_answer[0])


# Evaluator


@ICL_EVALUATORS.register_module()
class BFCLASTEvaluator(BaseEvaluator):
    """BFCL single-turn AST evaluator.

    Each reference is the JSON-encoded ``gold`` produced by ``BFCLDataset``,
    which packs the function descriptions, the possible answers and the
    category of each entry.
    """

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        details = []
        correct = 0
        for pred, reference in zip(predictions, references):
            if isinstance(pred, dict):
                pred = pred.get('prediction', '')
            gold = json.loads(reference)
            category = gold['category']
            functions = gold['functions']
            ground_truth = gold['ground_truth']

            valid = False
            error_type = ''
            decoded = None
            try:
                decoded = bfcl_ast_parse(pred)
                if not is_function_calling_format_output(decoded):
                    error_type = 'ast_decoder:decoder_wrong_output_format'
                else:
                    checker_result = bfcl_ast_checker(functions, decoded,
                                                      ground_truth, category)
                    valid = checker_result['valid']
                    if not valid:
                        error_type = checker_result['error_type']
            except Exception:
                error_type = 'ast_decoder:decoder_failed'

            details.append({
                'pred': pred,
                'answer': ground_truth,
                'correct': valid,
                'error_type': error_type,
                'decoded_result': decoded,
            })
            if valid:
                correct += 1

        result = {
            'accuracy': 100 * correct / len(details) if details else 0,
            'details': details,
        }
        return result
