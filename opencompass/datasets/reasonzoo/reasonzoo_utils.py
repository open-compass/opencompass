import json
import os
import re

import yaml


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


def evaluate_responses(data):
    results = []

    # --------------
    # Lightweight helpers adapted from working logic in eval_standard_utils.py
    # --------------
    def _normalize(s):
        return re.sub(r'[^a-z0-9]', '', str(s).lower())

    def _unwrap_once(s):
        patterns = [
            r'^\s*\\boxed\s*\{\s*(.*?)\s*\}\s*$',
            r'^\s*\[\[\s*(.*?)\s*\]\]\s*$',
            r'^\s*\[\s*(.*?)\s*\]\s*$',
        ]
        for pat in patterns:
            m = re.match(pat, s, flags=re.DOTALL)
            if m:
                return m.group(1)
        return s

    def _fully_unwrap(s):
        prev = None
        while prev != s:
            prev, s = s, _unwrap_once(s)
        return s.strip()

    def _clean_latex(latex_expr):
        if '=' in latex_expr:
            latex_expr = latex_expr.rsplit('=', 1)[1]
        latex_expr = re.sub(r'\\[()\[\]]', '', latex_expr)
        latex_expr = re.sub(r'\\text\{.*?\}', '', latex_expr)
        latex_expr = re.sub(r'\\(left|right|displaystyle)', '', latex_expr)
        latex_expr = latex_expr.replace('\\\\', '\\')
        latex_expr = re.sub(r'\s+', ' ', latex_expr).strip()
        return latex_expr

    def _extract_text_from_brackets(text, clean_level='basic'):
        text = '' if text is None else str(text)
        matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
        if not matches:
            matches = re.findall(r'\$\\boxed\{(.*?)\}\$', text, re.DOTALL)
        if not matches:
            matches = re.findall(r'\\boxed\{(.*?)\}', text, re.DOTALL)
        if not matches:
            matches = re.findall(r'\$\$\s*\\boxed\{(.*?)\}\s*\$\$', text,
                                 re.DOTALL)
        if not matches:
            matches = re.findall(r'\[\s*(.*?)\s*\]', text, re.DOTALL)
        if not matches:
            matches = re.findall(r'is\s*\*\*(.*?)\*\*', text, re.DOTALL)
        if not matches:
            matches = re.findall(r'FINAL ANSWER:\s*(.*?)(?:\n|$)', text,
                                 re.DOTALL)
        if not matches:
            matches = re.findall(r'```output\n(.*?)\n```', text, re.DOTALL)

        # Some numeric fallbacks in verbose texts
        if not matches:
            for pat in [
                    r'Maximum Profit:?\s*\$?([\d,\.]+)',
                    r'Total Profit:?\s*\$?([\d,\.]+)',
                    r'Profit:?\s*\$?([\d,\.]+)',
                    r'(?:result|answer|value|optimal|solution)'
                    r'(?:\s+is)?:?\s*\$?([\d,\.]+)'
            ]:
                matches = re.findall(pat, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    break

        if matches:
            match_str = matches[-1].strip()
            if clean_level == 'clean':
                match_str = (match_str.replace('"', '').replace(
                    '\n', '').replace(' ', '').replace('[', '').replace(
                        ']', '').replace('\\',
                                         '').replace("'",
                                                     '').replace(',', ' '))
            elif clean_level == 'logic':
                match_str = match_str.replace('"',
                                              '').replace('\n', '').replace(
                                                  ' ', '').replace('.', '')
            elif clean_level == 'math':
                match_str = match_str.replace('"', '').replace(
                    '\n', '').replace('[', '').replace(']',
                                                       '').replace('$', '')
                match_str = f'{_clean_latex(match_str)}'
            return match_str
        # As a last resort, return the cleaned latex of the whole text
        # if math-like
        if ('\\frac{' in text or '\\pi' in text or '\\left(' in text
                or '\\right)' in text):
            return _clean_latex(text)
        return text

    def _judge(response_text, answer_text):
        text = (response_text or '').replace('\n', ' ')
        combo_pat = (r'(\\boxed\s*\{\s*.*?\s*\})|(\[\[\s*.*?\s*\]\])|'
                     r'(\[\s*.*?\s*\])')
        matches = [(m.start(), m.group(0))
                   for m in re.finditer(combo_pat, text, flags=re.DOTALL)]
        if not matches:
            return _normalize(answer_text) in _normalize(text)
        last_raw = matches[-1][1]
        last_clean = _fully_unwrap(last_raw)
        target_clean = _fully_unwrap(answer_text or '')
        return _normalize(last_clean) == _normalize(target_clean)

    def _remove_non_alnum(s):
        return re.sub(r'\W+', '', str(s))

    def _extract_numbers(s_in):
        try:
            matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(s_in))
            return [float(num) for num in matches]
        except Exception:
            return None

    def _evaluate_one(response, answer, question_type, rule_id, idx):
        # Normalize inputs
        response = '' if response is None else str(response)
        answer = '' if answer is None else str(answer)

        # Specialized numeric tolerance for ops-research-like tasks
        if question_type in ['operation_research']:
            resp_nums = _extract_numbers(
                _extract_text_from_brackets(response, 'clean')) or []
            ans_clean = re.sub(r'[^0-9.]', '',
                               _extract_text_from_brackets(answer, 'clean'))
            try:
                ans_num = float(ans_clean) if ans_clean else None
            except Exception:
                ans_num = None
            if ans_num is None:
                return _judge(response, answer)
            for rn in resp_nums:
                if ans_num == 0:
                    if abs(rn - ans_num) < 0.001:
                        return True
                else:
                    if abs(rn - ans_num) / abs(ans_num) < 0.05:
                        return True
            return False

        # Logic-like tasks: relaxed normalization
        if question_type in ['logic_calculation']:
            r_txt = _extract_text_from_brackets(response, 'clean')
            a_txt = _extract_text_from_brackets(answer, 'clean')
            r_norm = re.sub(r'[^A-Za-z0-9]', '', r_txt).lower()
            a_norm = re.sub(r'[^A-Za-z0-9]', '', a_txt).lower()
            r_spec = _remove_non_alnum(
                response.replace('[[', '[').replace(']]', ']'))
            a_spec = _remove_non_alnum(
                answer.replace('[[', '[').replace(']]', ']'))
            if r_norm == a_norm or r_spec == a_spec:
                return True
            # Fallback: judge by final bracketed
            return _judge(response, answer)

        # General fallback: robust bracket/boxed matching
        return _judge(response, answer)

    # Support dict or list input
    if isinstance(data, dict):
        iterable = data.items()
    else:
        iterable = enumerate(data)

    for key, record in iterable:
        idx = key
        response = record.get('prediction', record.get('response', ''))
        question_type = record.get('category', '')
        answer = record.get('gold', record.get('answer', ''))
        rule_id = record.get('rule_id', '')
        is_correct = _evaluate_one(response, answer, question_type, rule_id,
                                   str(idx))
        result_dict = {
            'idx': idx,
            'response': response,
            'answer': answer,
            'is_correct': is_correct
        }
        results.append(result_dict)
    return results
