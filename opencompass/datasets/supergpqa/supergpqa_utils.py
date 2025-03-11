import json
import os
import re

import sympy as sp
import yaml
from sympy.parsing.latex import parse_latex


def load_yaml(yaml_path):
    """Load a YAML file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f'YAML file not found: {yaml_path}')
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_json_or_jsonl(file_path):
    """Load data from a JSON or JSONL file."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.json'):
            return json.load(file)
        elif file_path.endswith('.jsonl'):
            return [json.loads(line) for line in file]
    return None


def find_file(base_path, sub_path, extensions=('json', 'jsonl')):
    """Find the first available file with given extensions."""
    for ext in extensions:
        file_path = os.path.join(base_path, f'{sub_path}.{ext}')
        if os.path.exists(file_path):
            return file_path
    return None


def load_json_or_jsonl_with_idx(data_path, split='', idx=None):
    base_path = os.path.join(data_path, split)
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    elif base_path.endswith('.json') or base_path.endswith('.jsonl'):
        file_path = base_path
    else:
        raise FileNotFoundError('No JSON or JSONL file found.')

    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.json'):
            data = json.load(file)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in file]

    if idx is not None:
        try:
            return next(item for item in data if item.get('idx') == idx)
        except StopIteration:
            raise ValueError(f'No entry found for idx {idx}')
    else:
        return data


def load_split_data(base_path, split_name):
    """Load the rule and sample data for a specific split."""
    split_path = os.path.join(base_path, split_name)
    rule_path = find_file(split_path, 'rule')
    sample_path = find_file(split_path, 'sample')

    rules = load_json_or_jsonl(rule_path) if rule_path else []
    samples = load_json_or_jsonl(sample_path) if sample_path else []

    return {'rules': rules, 'samples': samples}


def process_mixed_data(base_path, mode):
    """Load and process data for the 'mixed' split and specific mode."""
    mixed_path = os.path.join(base_path, 'mixed')
    file_path = find_file(mixed_path, mode)
    if not file_path:
        print(f'[WARNING] Missing file for mixed mode: {mode}')
        return []

    data = load_json_or_jsonl(file_path)
    template_path = os.path.join(base_path, 'config/prompt/mixed.yaml')
    template = load_yaml(template_path)

    processed = []
    for item in data:
        rules = '\n'.join(item.get('rule_list', []))
        questions = '\n'.join(item.get('question_list', []))
        item['prompt'] = template['prompt_format'][0].format(rules, questions)
        processed.append(item)

    return processed


class ConfigWrapper:

    def __init__(self, config_path):
        self._config = {}
        with open(config_path, 'r') as file:
            self._config = yaml.safe_load(file)
        for key, value in self._config.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._config[key] = value
            super().__setattr__(key, value)

    def __getattr__(self, key):
        if key in self._config:
            return self._config[key]
        raise AttributeError(
            f"'ConfigWrapper' object has no attribute '{key}'")

    def get_id(self, data):
        if isinstance(self._config.get('id_key'), str):
            return data.get(self._config.get('id_key'), None)
        elif isinstance(self._config.get('id_key'), list):
            return '_'.join([
                str(data[key]) for key in self._config.get('id_key')
                if key in data
            ])

    def print_all_keys(self):
        print('config keys:')
        for key, value in self._config.items():
            print(f'  - {key}: {value}')


config_wrapper = None


def initialize_config(config_path):
    global config_wrapper
    config_wrapper = ConfigWrapper(config_path)


def get_config_wrapper():
    global config_wrapper
    if config_wrapper is None:
        raise RuntimeError(
            'ConfigWrapper not initialized. Call initialize_config first.')
    return config_wrapper


if __name__ == '__main__':
    config_path = 'config/config.yaml'
    initialize_config(config_path)
    data = {
        'idx':
        '50',
        'step':
        21,
        'question':
        ('Ciphertext: "17,156,4,54,213,17,23,84,228,54,281"\n\n'
         'Please provide the decrypted answer, encapsulated in double '
         'square brackets. '
         'For example, the format should be: [[decrypted answer]].'),
        'answer':
        '[[P]]',
        'category':
        'Decryption',
        'rule_id':
        '23',
        'input':
        'Ciphertext: "17,156,4,54,213,17,23,84,228,54,281"',
        'steps_num':
        23,
        'description':
        ('For a number c=228 in the ciphertext:\n'
         'Calculate z = c^e mod n. Here ^ means multiplication.\n'
         'z is 80.\nBased on the decimal number represented by z, '
         'use the ascii code to find the corresponding letter '
         'as the plaintext letter p.\n'
         'Please give the letter p in [[...]] format.\n'),
        'atom':
        80
    }
    print(config_wrapper.get_id(data))


def read_yaml(config='default'):
    if os.path.exists(f'config/prompt/{config}.yaml'):
        yaml_file = f'config/prompt/{config}.yaml'
    else:
        yaml_file = config
    with open(yaml_file, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def write_jsonl_lines(file, data):
    config_wrapper = get_config_wrapper()
    if config_wrapper.save_prompt:
        json.dump(data, file, ensure_ascii=False)
    else:
        data.pop(config_wrapper.prompt_key)
        json.dump(data, file, ensure_ascii=False)
    file.write('\n')
    file.flush()


def print_info(info):
    print('-' * 100)
    print('[INFO] model_name:', info['model_name'])
    print('[INFO] splits:', info['splits'])
    print('[INFO] modes:', info['modes'])
    print('[INFO] output_dir:', info['output_dir'])
    print('[INFO] Infer Limit:',
          'No limit' if info['infer_limit'] is None else info['infer_limit'])
    print('[INFO] Number of Workers:', info['num_workers'])
    print('[INFO] Batch Size:', info['batch_size'])
    print('[INFO] Use Accel:', info['use_accel'])
    print('-' * 100)


def read_json_or_jsonl(data_path, split='', mapping_key=None):
    base_path = os.path.join(data_path, split)
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    elif base_path.endswith('.json') or base_path.endswith('.jsonl'):
        file_path = base_path
    else:
        raise FileNotFoundError('No JSON or JSONL file found.')

    with open(file_path, 'r') as file:
        if file_path.endswith('.json'):
            data = json.load(file)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in file]

    if mapping_key:
        return {
            item[mapping_key]: item
            for item in data if mapping_key in item
        }
    else:
        return data


def read_json_or_jsonl_with_idx(data_path, split='', idx=None):
    base_path = os.path.join(data_path, split)
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    elif base_path.endswith('.json') or base_path.endswith('.jsonl'):
        file_path = base_path
    else:
        raise FileNotFoundError('No JSON or JSONL file found.')

    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.json'):
            data = json.load(file)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in file]

    if idx is not None:
        try:
            return next(item for item in data if item.get('idx') == idx)
        except StopIteration:
            raise ValueError(f'No entry found for idx {idx}')
    else:
        return data


idx_ranges = [
    [18],
    [73, 74, 77],
    [94],
    [115, 116, 117],
    [121, 122, 123, 125],
    [131, 132, 134, 135, 136],
    [141, 143, 149],
    list(range(145, 148)),
    list(range(151, 157)),
    [160, 161, 162],
    [164, 165, 166],
    [170],
    [206, 209],
    list(range(211, 216)),
    [217, 218],
]


def clean_json_string(json_str):
    json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
    return json_str


def is_in_idx_ranges(idx, idx_ranges):
    for range_list in idx_ranges:
        if int(idx) in range_list:
            return True
    return False


def extract_json(text):
    matches = re.findall(r'{.*}', text, re.DOTALL)
    if matches:
        json_str = matches[-1]
        json_str = clean_json_string(json_str)
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print(f'Error decoding JSON: {e}')
            return 'NULL'
    return 'NULL'


def extract_all_responses_from_json(response_json):
    results = []
    for key, value in response_json.items():
        results.append(str(value))
    return results


def clean_latex(latex_expr):
    if '=' in latex_expr:
        latex_expr = latex_expr.rsplit('=', 1)[1]
    latex_expr = re.sub(r'\\[()\[\]]', '', latex_expr)
    latex_expr = re.sub(r'\\text\{.*?\}', '', latex_expr)
    latex_expr = re.sub(r'\\(left|right|displaystyle)', '', latex_expr)
    latex_expr = latex_expr.replace('\\\\', '\\')
    return latex_expr


def extract_text_from_brackets(text, clean_level='basic'):
    matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
    if not matches:
        matches = re.findall(r'\$\\boxed\{(.*?)\}\$', text, re.DOTALL)
    if not matches:
        matches = re.findall(r'\[\s*(.*?)\s*\]', text, re.DOTALL)
    if matches:
        match_str = matches[0].strip()
        if clean_level == 'clean':
            match_str = match_str.replace('"', '').replace('\n', '').replace(
                ' ', '').replace('[', '').replace(']', '')
        elif clean_level == 'logic':
            match_str = match_str.replace('"', '').replace('\n', '').replace(
                ' ', '').replace('.', '')
        elif clean_level == 'math':
            match_str = match_str.replace('"', '').replace('\n', '').replace(
                '[', '').replace(']', '').replace('$', '')
            return f'{clean_latex(match_str)}'
        return f'[[{match_str}]]'
    return 'NULL'


def extract_inner_text_from_brackets(text):
    if not isinstance(text, str):
        print(f'text type: {type(text)}, text value: {text}')
        return 'NULL'
    match = re.search(r'\[\[(.*?)\]\]', text, re.DOTALL)
    return match.group(1) if match else 'NULL'


def extract_numbers(str):
    numbers = re.findall(r'\d+', str)
    numbers = list(map(int, numbers))
    return numbers


def extract_and_sort_inequalities(latex_expr):
    pattern = r'(≥|≤)\s*([-]?\d+\.?\d*)'
    matches = re.findall(pattern, latex_expr)
    extracted_inequalities = [''.join(match) for match in matches]
    sorted_inequalities = sorted(extracted_inequalities)
    return sorted_inequalities


def rule5_normalize_content(content):
    parts = [part for part in content.split(';')]
    sorted_parts = sorted(parts)
    return sorted_parts


def normalize_string(s):
    s = re.sub(r'[^0-9]', '', s)
    pairs = s.split(',')
    pairs.sort()
    return pairs


def remove_commas_and_spaces(s):
    return re.sub(r'[,\s\[\]]+', '', s)


def remove_non_alphanumeric(s):
    return re.sub(r'\W+', '', s)


def contains_or(answer):
    return 'or' in answer


def compare_multi_results(response, answer):
    try:
        response_text = extract_text_from_brackets(response, 'clean')
        response_text = re.sub(r'\\text\{or\}', 'or', response_text)
        if response_text == 'NULL':
            return False
        answer = extract_text_from_brackets(answer, 'clean')
        response_split = response_text.strip('[[]]').split('or')
        answer_split = answer.strip('[[]]').split('or')
        response_sorted = sorted([x.strip() for x in response_split])
        answer_sorted = sorted([x.strip() for x in answer_split])
        return response_sorted == answer_sorted
    except Exception as e:
        print(f'Error during comparison: {e}')
        return False


def split_or_expression(expression):
    return [part.strip() for part in expression.split('or')]


def compare_math_expressions(response, answer):
    response_text = extract_text_from_brackets(response, 'math')
    answer_text = extract_text_from_brackets(answer, 'math')
    if response_text == 'NULL':
        return False
    if contains_or(answer_text):
        response_parts = split_or_expression(response_text)
        answer_parts = split_or_expression(answer_text)
        try:
            response_exprs = {
                sp.simplify(parse_latex(part))
                for part in response_parts
            }
            answer_exprs = {
                sp.simplify(parse_latex(part))
                for part in answer_parts
            }
            return response_exprs == answer_exprs
        except Exception as e:
            print(f'Error during simplification or parsing: {e}')
            return response_text == answer_text
    else:
        try:
            response_expr = sp.simplify(parse_latex(response_text))
            answer_expr = sp.simplify(parse_latex(answer_text))
            return response_expr == answer_expr
        except Exception as e:
            print(f'Error during simplification or parsing: {e}')
            return response_text == answer_text


def method_equal(response_text, answer):
    return response_text == answer


def method_1(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    answer = re.sub(r'[^A-Za-z]', '', answer)
    answer = answer.lower()
    return cleaned_string == answer


def method_2(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    answer = answer.split(',')
    return cleaned_string in answer


def method_3(response_text, answer):
    response_text = response_text.lower()
    pairs1 = re.split(r'\W+', response_text)
    pairs2 = answer.split(' ')
    pairs1 = [word for word in pairs1 if word]
    pairs1.sort()
    pairs2.sort()
    return pairs1 == pairs2


def method_4(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    return cleaned_string in answer


def method_5(response_text, answer):
    response_text = re.sub(r'\s+', '', response_text)
    response_text = response_text.split(',')
    answer = answer.split(',')
    response_text.sort()
    answer.sort()
    return response_text == answer


def method_9(response_text, answer):
    response_text = response_text.replace('×', '*').replace('−', '-')
    answer = answer.replace('×', '*').replace('−', '-')

    def extract_operators(s):
        return re.findall(r'[+\-*/]', s)

    response_ops = extract_operators(response_text.split('=')[0])
    answer_ops = extract_operators(answer.split('=')[0])
    if response_ops != answer_ops:
        return False
    match = re.search(r'=\s*(-?\d+)', answer)
    expected_result = int(match.group(1))
    try:
        left_side = response_text.split('=')[0]
        result = eval(left_side)
    except Exception as e:
        print(f'Error during evaluation: {e}')
        return False
    return result == expected_result


def method_10(response_text, answer):
    response_text = response_text.replace('×', '*').replace('−', '-')
    response_text = response_text.split('=')[0]
    answer = answer.split('\n')[0].split('=')[0]
    response_ops = sorted(remove_non_alphanumeric(response_text))
    answer_ops = sorted(remove_non_alphanumeric(answer))
    if response_ops != answer_ops:
        return False
    try:
        result = eval(response_text)
    except Exception as e:
        print(f'Error during evaluation: {e}')
        return False
    return result == 24


def method_18(response_text, answer):
    cleaned_s1 = remove_commas_and_spaces(response_text)
    cleaned_s2 = remove_commas_and_spaces(answer)
    return cleaned_s1 == cleaned_s2


def method_general(response_text, answer):
    cleaned_s1 = remove_non_alphanumeric(response_text)
    cleaned_s2 = remove_non_alphanumeric(answer)
    return cleaned_s1 == cleaned_s2


question_methods = {
    '1': method_1,
    '2': method_2,
    '3': method_3,
    '4': method_4,
    '5': method_5,
    '9': method_9,
    '10': method_10,
    '18': method_18,
}


def evaluate_response_vs_answer(response, answer, question_type, rule_id, idx):
    if question_type == 'logic' and rule_id == '5':
        response_text = extract_text_from_brackets(response, 'logic')
        answer_text = extract_text_from_brackets(answer, 'logic')
        if response_text is None:
            return False
        normalized_response = rule5_normalize_content(response_text)
        normalized_answer = rule5_normalize_content(answer)
        return normalized_response == normalized_answer
    elif question_type == 'logic':
        response_text = extract_text_from_brackets(response, 'logic')
        answer_text = extract_text_from_brackets(answer, 'logic')
        return response_text == answer_text
    elif question_type == 'operation' and (idx == '178' or idx == '179'):
        response_text = extract_text_from_brackets(response, 'clean')
        response_text = extract_and_sort_inequalities(response_text)
        answer_text = extract_and_sort_inequalities(answer)
        # print(response_text, answer_text)
        return response_text == answer_text
    elif question_type == 'operation' and rule_id == '18':
        response_text = extract_text_from_brackets(response, 'clean')
        answer = extract_inner_text_from_brackets(answer)
        response_text = ''.join(sorted(re.sub(r'\W+', '', response_text)))
        answer = ''.join(sorted(re.sub(r'\W+', '', answer)))
        return response_text == answer
    elif question_type == 'operation' and rule_id in {'23', '24', '25'}:
        response_text = extract_text_from_brackets(response, 'clean')
        if response_text is None:
            return False
        response_text = extract_numbers(response_text)
        answer_text = extract_numbers(answer)
        return response_text == answer_text
    elif question_type == 'operation' and is_in_idx_ranges(idx, idx_ranges):
        return compare_math_expressions(response, answer)
    elif question_type == 'operation' and contains_or(answer):
        return compare_multi_results(response, answer)
    elif question_type == 'puzzle':
        response_text = extract_inner_text_from_brackets(response)
        answer = extract_inner_text_from_brackets(answer)
        method = question_methods.get(rule_id)
        if method:
            return method(response_text, answer)
        return method_general(response_text, answer)
    else:
        response_text = extract_text_from_brackets(response, 'clean')
        return response_text == answer


def compute_one_mixed_question_pass_rate(idx,
                                         question_list,
                                         response_json,
                                         base_path=None):
    if response_json == 'NULL':
        result_dict = {
            'idx': idx,
            'response': response_json,
            'details': None,
            'pass_rate': 0,
            'is_correct': False
        }
        return result_dict
    response_list = extract_all_responses_from_json(response_json)
    correct_num = 0
    results = []
    for q_idx, question in enumerate(question_list):
        category, question_idx = question.rsplit('_', 1)
        question_content = load_json_or_jsonl_with_idx(base_path,
                                                       os.path.join(
                                                           category, 'sample'),
                                                       idx=question_idx)
        answer = question_content['answer']
        if q_idx >= len(response_list):
            break
        response = response_list[q_idx]
        response_text = extract_text_from_brackets(response)
        rule_id = question_content['rule_id']
        is_correct = evaluate_response_vs_answer(response, answer, category,
                                                 rule_id, q_idx)
        if is_correct:
            correct_num += 1
        results.append({
            'question': question,
            'response_text': response_text,
            'answer': answer,
            'is_correct': is_correct
        })

    pass_rate = correct_num / len(question_list)
    question_correct = pass_rate == 1.0
    result_dict = {
        'idx': idx,
        'response': response_json,
        'details': results,
        'pass_rate': pass_rate,
        'is_correct': question_correct
    }
    return result_dict


def evaluate_responses(data, mode, base_path=None):
    results = []

    # Iterate over the values of the dictionary (numerical keys)
    for key, record in data.items():
        idx = key  # Use the dictionary key as the "idx"
        response = record.get('prediction', '')
        question_type = record.get('category', '')
        response_text = extract_text_from_brackets(response)
        answer = record.get('gold', '')
        rule_id = record.get('rule_id', '')
        is_correct = evaluate_response_vs_answer(response, answer,
                                                 question_type, rule_id, idx)
        result_dict = {
            'idx': idx,
            'response': response,
            'response_text': response_text,
            'answer': answer,
            'is_correct': is_correct
        }
        if question_type == 'counterfactual':
            real_life_answer = record.get('real_life_answer', '')
            is_real_life = evaluate_response_vs_answer(response,
                                                       real_life_answer,
                                                       question_type, rule_id,
                                                       idx)
            result_dict['real_life_answer'] = real_life_answer
            result_dict['is_real_life'] = is_real_life
        if question_type == 'cipher' and mode == 'subquestions':
            result_dict['type'] = record.get('type', '')
        results.append(result_dict)
    return results
