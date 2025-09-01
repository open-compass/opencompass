import json
import re
import sympy as sp
from sympy.parsing.latex import parse_latex
import os 
from utils.common import read_json_or_jsonl_with_idx
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import time
import subprocess
import tempfile
import ast
from openai import OpenAI
import hashlib
import sys
import requests
from black import format_str, FileMode
import random

# ---------- Improved evaluation functions ----------
def _normalize(s: str) -> str:
    """大小写统一、去掉非字母数字，方便宽松比对"""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def _unwrap_once(s: str) -> str:
    """去掉最外层的 [[]] / [] / \boxed{}，若有则返回内部内容"""
    patterns = [
        r'^\s*\\boxed\s*{\s*(.*?)\s*}\s*$',
        r'^\s*\[\[\s*(.*?)\s*\]\]\s*$',
        r'^\s*\[\s*(.*?)\s*\]\s*$'
    ]
    for pat in patterns:
        m = re.match(pat, s, flags=re.DOTALL)
        if m:
            return m.group(1)
    return s

def _fully_unwrap(s: str) -> str:
    """递归剥掉所有包裹层"""
    prev = None
    while prev != s:
        prev, s = s, _unwrap_once(s)
    return s.strip()

def judge(response_text: str, answer_text: str) -> bool:
    """改进的评测逻辑"""
    # 1. 拉平换行，便于一次性搜索
    text = response_text.replace('\n', ' ')

    # 2. 统一的"候选答案"正则：三种形式任选其一（使用非贪婪 .*? 并允许跨行）
    combo_pat = r'(\\boxed\s*{\s*.*?\s*})|(\[\[\s*.*?\s*\]\])|(\[\s*.*?\s*\])'
    
    # 3. 记录所有匹配（位置 + 内容）
    matches = [(m.start(), m.group(0)) for m in re.finditer(combo_pat, text, flags=re.DOTALL)]
    if not matches:                       # 若没任何候选，走旧的兜底逻辑
        return _normalize(answer_text) in _normalize(text)

    # 4. 取"最后出现"的那一段，彻底剥壳得到纯内容
    last_raw = matches[-1][1]
    last_clean = _fully_unwrap(last_raw)

    # 5. 把官方答案也做同样的剥壳与正规化后比较
    target_clean = _fully_unwrap(answer_text)
    return _normalize(last_clean) == _normalize(target_clean)

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
            return "NULL"
    return "NULL"

def extract_all_responses_from_json(response_json):
    results=[]
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
    # Add more normalization for common LaTeX expressions
    latex_expr = re.sub(r'\s+', ' ', latex_expr).strip()
    return latex_expr

def extract_text_from_brackets(text, clean_level="basic"):
    # Existing patterns
    matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
    if not matches:
        matches = re.findall(r'\$\\boxed\{(.*?)\}\$', text, re.DOTALL)
    if not matches:
        # Also try to match \boxed without the $ delimiters
        matches = re.findall(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    if not matches:
        # Add pattern to match $$\n\boxed{...}\n$$ format (double dollar signs with newlines)
        matches = re.findall(r'\$\$\s*\\boxed\{(.*?)\}\s*\$\$', text, re.DOTALL)
    if not matches:
        matches = re.findall(r'\[\s*(.*?)\s*\]', text, re.DOTALL)
    if not matches:
        matches = re.findall(r'is\s*\*\*(.*?)\*\*', text, re.DOTALL)
    if not matches:
        # Add pattern to match "FINAL ANSWER: [content]"
        matches = re.findall(r'FINAL ANSWER:\s*(.*?)(?:\n|$)', text, re.DOTALL)
    if not matches:
        # ```output\n[content]\n```
        matches = re.findall(r'```output\n(.*?)\n```', text, re.DOTALL)
    # New patterns to detect more answer formats
    if not matches:
        # Look for "Maximum Profit: $X" or similar profit statements
        matches = re.findall(r'Maximum Profit:?\s*\$?([\d,\.]+)', text, re.DOTALL | re.IGNORECASE)
    if not matches:
        # Look for "Total Profit: $X" pattern
        matches = re.findall(r'Total Profit:?\s*\$?([\d,\.]+)', text, re.DOTALL | re.IGNORECASE)
    if not matches:
        # Look for "Profit: $X" pattern
        matches = re.findall(r'Profit:?\s*\$?([\d,\.]+)', text, re.DOTALL | re.IGNORECASE)
    if not matches:
        # Catch most numeric results with currency symbols
        matches = re.findall(r'(?:result|answer|value|optimal|solution)(?:\s+is)?:?\s*\$?([\d,\.]+)', text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        match_str = matches[-1].strip()
        if clean_level == "clean": 
            # Preserve number separators by replacing commas with spaces
            match_str = (match_str.replace('"', '')
                         .replace('\n', '')
                         .replace(' ', '')
                         .replace('[', "")
                         .replace(']', "")
                         .replace('\\', '')
                         .replace("'", "")
                         .replace(',', ' '))  # Change comma replacement to space
        elif clean_level == "logic":  
            match_str = match_str.replace('"', '').replace('\n', '').replace(' ', '').replace('.', "")
        elif clean_level == "math":
            match_str = match_str.replace('"', '').replace('\n', '').replace('[', "").replace(']', "").replace('$',"")
            # Don't immediately return here, continue with normal flow
            match_str = f'{clean_latex(match_str)}'
        elif 'ANSWER:' in text:
            match_str = text.split('ANSWER:')[1].strip()
        return match_str
    
    # If no brackets found but text contains math expression, try to extract it directly
    if '\\frac{' in text or '\\pi' in text or '\\left(' in text or '\\right)' in text:
        return clean_latex(text)
    
    return text

def extract_inner_text_from_brackets(text):
    if not isinstance(text, str):
        print(f"text type: {type(text)}, text value: {text}")
        return "NULL"
    match = re.search(r'\[\[(.*?)\]\]', text, re.DOTALL)
    return match.group(1) if match else "NULL"


def extract_numbers(s_in: str):
    """
    Parses a string and extracts all valid floating point numbers.
    """
    try:
        # Use a regex that properly identifies float numbers
        # This will match numbers like 123, -123, 0.123, -0.123, .123, -.123
        matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s_in)
        
        # Convert all matches to float
        numbers = [float(num) for num in matches]
        return numbers
    except (ValueError, SyntaxError, TypeError) as e:
        # Handle cases where conversion fails
        print(f"Error: Input string '{s_in}' contains invalid number formats. Details: {e}")
        return None


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
    pairs = s.split(",")
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
        response_text = extract_text_from_brackets(response,"clean")
        response_text = re.sub(r'\\text\{or\}', 'or', response_text)
        if response_text == "NULL":
            return False
        answer=extract_text_from_brackets(answer,"clean")
        response_split = response_text.strip('[[]]').split('or')
        answer_split = answer.strip('[[]]').split('or')
        response_sorted = sorted([x.strip() for x in response_split])
        answer_sorted = sorted([x.strip() for x in answer_split])
        return response_sorted == answer_sorted
    except Exception as e:
        return False

def split_or_expression(expression):
    return [part.strip() for part in expression.split('or')]

def compare_math_expressions(response, answer):
    """Compare mathematical expressions with better handling of common formats."""
    response_text = extract_text_from_brackets(response, "math")
    
    # Try multiple variants of answer extraction
    answer_variants = [
        extract_text_from_brackets(answer, "math"),
        answer.strip(),
        re.sub(r'\\left\(|\\\right\)', '', answer).strip()
    ]
    
    # Try direct symbolic comparison first
    try:
        resp_expr = sp.sympify(response_text.replace('\\', '').replace('frac', '').replace('pi', 'Pi'))
        for ans_text in answer_variants:
            try:
                ans_expr = sp.sympify(ans_text.replace('\\', '').replace('frac', '').replace('pi', 'Pi'))
                if sp.simplify(resp_expr - ans_expr) == 0:
                    return True
            except:
                continue
    except:
        pass
    
    # Try string normalization comparison
    norm_resp = normalize_math_expression(response_text)
    for ans_text in answer_variants:
        norm_ans = normalize_math_expression(ans_text)
        if norm_resp == norm_ans:
            return True
    
    # Special case for ordered pairs like (3, pi/2)
    pair_pattern = r'\(([^,]+),([^)]+)\)'
    resp_match = re.search(pair_pattern, response_text)
    
    if resp_match:
        resp_parts = [resp_match.group(1).strip(), resp_match.group(2).strip()]
        
        for ans_text in answer_variants:
            ans_match = re.search(pair_pattern, ans_text)
            if ans_match:
                ans_parts = [ans_match.group(1).strip(), ans_match.group(2).strip()]
                
                # Try to compare each part
                parts_match = True
                for i in range(2):
                    try:
                        r_expr = sp.sympify(resp_parts[i].replace('\\', '').replace('frac', '').replace('pi', 'Pi'))
                        a_expr = sp.sympify(ans_parts[i].replace('\\', '').replace('frac', '').replace('pi', 'Pi'))
                        if sp.simplify(r_expr - a_expr) != 0:
                            parts_match = False
                            break
                    except:
                        if normalize_math_expression(resp_parts[i]) != normalize_math_expression(ans_parts[i]):
                            parts_match = False
                            break
                
                if parts_match:
                    return True
    
    return False

def method_equal(response_text, answer):
    return response_text==answer

def method_1(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    answer=re.sub(r'[^A-Za-z]', '', answer)
    answer= answer.lower()
    return cleaned_string == answer

def method_2(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    answer=answer.split(",")
    return cleaned_string in answer

def method_3(response_text, answer):
    response_text = response_text.lower()
    pairs1 = re.split(r'\W+', response_text)
    pairs2=answer.split(" ")
    pairs1 = [word for word in pairs1 if word]
    pairs1.sort()
    pairs2.sort()
    return pairs1==pairs2

def method_4(response_text, answer):
    cleaned_string = re.sub(r'[^A-Za-z]', '', response_text)
    cleaned_string = cleaned_string.lower()
    return cleaned_string in answer

def method_5(response_text, answer):
    response_text=re.sub(r'\s+', '', response_text)
    response_text=response_text.split(",")
    answer=answer.split(",")
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
        return False
    return result == expected_result

def method_10(response_text, answer):
    response_text = response_text.replace('×', '*').replace('−', '-')
    response_text=response_text.split('=')[0]
    answer=answer.split('\n')[0].split('=')[0]
    response_ops = sorted(remove_non_alphanumeric(response_text))
    answer_ops = sorted(remove_non_alphanumeric(answer))
    if response_ops != answer_ops:
        return False
    try:
        result = eval(response_text)
    except Exception as e:
        if response_text == answer:
            return True
        else:
            return False
    return result==24

def method_18(response_text, answer):
    cleaned_s1 = remove_commas_and_spaces(response_text)
    cleaned_s2 = remove_commas_and_spaces(answer)
    return cleaned_s1 == cleaned_s2

def method_general(response_text, answer):
    cleaned_s1 = remove_non_alphanumeric(response_text)
    cleaned_s2 = remove_non_alphanumeric(answer)
    return cleaned_s1 == cleaned_s2

question_methods = {
    '1':method_1,
    '2':method_2,
    '3': method_3,
    '4':method_4, 
    '5': method_5, 
    '9':method_9, 
    '10': method_10, 
    '18':method_18,
}


def remove_special_characters(s):
    sub_pre = s.replace('[[', '[').replace(']]', ']')
    sub_pre = re.sub(r'[^A-Za-z0-9\[\]]', '', sub_pre)
    return sub_pre


def evaluate_response_vs_answer_without_python_code(response, answer, question_type, rule_id, idx):
    if  question_type == 'number_calculation':
        response_text = extract_text_from_brackets(response,"clean")
        # Normalize hexadecimal case before comparison
        response_text = response_text.lower()
        numbers_in_response = extract_numbers(response_text)
        
        answer = answer.lower()
        # if no . in answer string, then we can use the method_18
        if not '.' in answer:
            return method_18(response_text, answer)
        numbers_in_answer = extract_numbers(answer)
        
        # for each number in the response, check if it is within 1% of the number in the answer
        if numbers_in_response is None or numbers_in_answer is None:
            return False
        if len(numbers_in_response) != len(numbers_in_answer):
            return False

        for i in range(len(numbers_in_response)):
            if numbers_in_answer[i] == 0:
                if abs(numbers_in_response[i] - numbers_in_answer[i]) > 0.1:
                    return False
            elif abs(numbers_in_response[i] - numbers_in_answer[i]) / numbers_in_answer[i] > 0.1:
                return False
        return True
    
    
    elif question_type == 'puzzle_and_code' and rule_id == '8':
        # split by ,
        answer = extract_text_from_brackets(answer,"clean")
        answer_split = answer.split(',')
        normalized_response = re.sub(r'[^A-Za-z0-9]', '', response).lower()
        normalized_answer = re.sub(r'[^A-Za-z0-9]', '', answer).lower()
        for i in range(len(answer_split)):
            if answer_split[i] not in response:
                return False
        return True
    elif question_type == 'puzzle_and_code' and rule_id == '10':
        # eval the 24 game
        response_text = extract_text_from_brackets(response,"clean")
        answer_text = extract_text_from_brackets(answer,"clean")
        return method_10(response_text, answer_text)
    elif question_type == 'formal_language':
        response_nums = re.findall(r't\d+', response)
        answer_nums = re.findall(r't\d+', answer)
        return response_nums and answer_nums and response_nums[-1] in answer_nums
    
    elif question_type in ['operation_research', 'puzzle_and_code', 'cipher_and_code', 'zebra']:
        response_text = extract_text_from_brackets(response, "clean")
        answer_text = extract_text_from_brackets(answer, "clean")
        
        # Look for profit values in the response if the answer is numeric
        if answer_text.replace('.', '').isdigit():
            # First try direct numeric comparison if both are numeric
            response_clean = re.sub(r'[^0-9.]', '', response_text)
            answer_clean = re.sub(r'[^0-9.]', '', answer_text)
            
            # Try to extract numbers from both
            response_numbers = extract_numbers(response_text)
            answer_numbers = extract_numbers(answer_text)
            
            # Also look for profit statements in the full response
            profit_matches = re.findall(r'(?:profit|result|value)(?:\s*is)?:?\s*\$?([\d,\.]+)', 
                                        response.lower(), re.IGNORECASE)
            
            # Convert profit matches to floats if found
            profit_values = []
            for match in profit_matches:
                try:
                    profit_values.append(float(match.replace(',', '')))
                except ValueError:
                    continue
                
            # Use all extracted numbers for comparison
            all_response_numbers = []
            if response_numbers:
                all_response_numbers.extend(response_numbers)
            if profit_values:
                all_response_numbers.extend(profit_values)
            
            # Try numeric comparison with relative tolerance for larger values
            try:
                answer_num = float(answer_clean)
                # Check if any extracted number matches with tolerance
                for resp_num in all_response_numbers:
                    # Use 5% relative tolerance for large numbers
                    if question_type == 'operation_research':
                        if answer_num == 0:
                            if abs(resp_num - answer_num) < 0.001:
                                return True
                        else:
                            if abs(resp_num - answer_num)/answer_num < 0.05:
                                    return True
                    else:
                        if answer_num > 100:
                            if abs(resp_num - answer_num)/answer_num < 0.05:
                                return True
                        else:
                            if abs(resp_num - answer_num) < 0.1:
                                return True
                return False
            except ValueError:
                pass
        
        # Fall back to improved text comparison for non-numeric answers
        return judge(response_text, answer_text)
    elif question_type == 'logic_calculation':
        response_text = extract_text_from_brackets(response,"clean")
        answer_text = extract_text_from_brackets(answer,"clean")
        normalized_response = re.sub(r'[^A-Za-z0-9]', '', response_text).lower()
        normalized_answer = re.sub(r'[^A-Za-z0-9]', '', answer_text).lower()
        normalized_response_special = remove_special_characters(str(response))
        normalized_answer_special = remove_special_characters(str(answer))
        number_normallized_answer_special = re.sub(r'[^0-9]', '', normalized_answer_special)
        number_normallized_response_special = re.sub(r'[^0-9]', '', normalized_response_special)
        if normalized_answer == normalized_response or normalized_answer_special == normalized_response_special or number_normallized_answer_special == number_normallized_response_special:
            return True
        else:
            return False
        
    elif question_type == 'cardgame':
        response_text = extract_text_from_brackets(response,"clean")
        answer_text = extract_text_from_brackets(answer,"clean")
        if "[[]]" in answer_text or "[[]]" in answer:
            return "[[]]" in response_text or "[[]]" in response
        # if the answer_text is digit or alphabetical characters, then check if the response_text is digit or alphabetical characters
        # if space is in answer_text, then replace it with ''
        answer_text = answer_text.replace(' ', '')
        response_text = response_text.replace(' ', '')
        if answer_text.isalnum():
            # filter all alphabetical characters and numeric characters
            response_text = re.sub(r'[^A-Za-z0-9]', '', response_text)
            answer_text = re.sub(r'[^A-Za-z0-9]', '', answer_text)
            return response_text.lower() == answer_text.lower()
        
        else:
            return answer_text in response_text or str(answer_text) in str(response_text) or str(answer) in str(response)
    
    # Convert to string first to ensure type safety
    response = str(response)
    answer = str(answer)
    
    # Extract response text (looking for double brackets, boxed content, or FINAL ANSWER)
    response_text = extract_text_from_brackets(response, "clean")
    
    # Clean up additional formatting characters like asterisks that might appear at the end
    response_text = re.sub(r'\*+$', '', response_text)
    # Clean up LaTeX box formatting if present
    response_text = re.sub(r'\\boxed{(.*?)}', r'\1', response_text)
    
    # Now apply lowercase if the response and answer are not purely numeric
    response = response.lower() if not response.isdigit() else response
    answer = answer.lower() if not answer.isdigit() else answer
    response_text = response_text.lower()
    
    # Remove all non-alphanumeric characters
    clean_response = re.sub(r'[^A-Za-z0-9]', '', response)
    clean_answer = re.sub(r'[^A-Za-z0-9]', '', answer)
    clean_response_text = re.sub(r'[^A-Za-z0-9]', '', response_text)
    
    # Check numeric values with tolerance
    if clean_response_text.isdigit() and clean_answer.isdigit():
        return abs(float(clean_response_text) - float(clean_answer)) < 0.001
        
    # For multiple choice, check if the response_text is in the answer
    if clean_answer.isalpha() and clean_response_text.isalpha():
        return clean_response_text == clean_answer
    normalized_response_special = remove_special_characters(str(response))
    normalized_answer_special = remove_special_characters(str(answer))
    # Final fallback comparison
    return clean_response == clean_answer or clean_response_text == clean_answer or normalized_answer_special in normalized_response_special

def remove_python_code_snippets(response):
    python_code_snippets = extract_python_scripts(response)
    for code in python_code_snippets:
        response = response.replace(code, "")
    return response

def prettify(code: str) -> str:
    """Format python code using black."""
    try:
        return format_str(code, mode=FileMode()).strip()
    except Exception as e:
        print(f"Warning: Black formatting failed: {e}. Using original code.", file=sys.stderr)
        return code

def execute_python_code(code: str, sandbox_url: str, timeout: float = 30.0) -> tuple[str, int]:
    """
    Executes the provided Python code via a remote sandbox HTTP API.
    """
    formatted_code = prettify(code)
    digest = hashlib.sha256(formatted_code.encode()).hexdigest()

    try:
        res = requests.post(
            sandbox_url,
            json={"code": formatted_code, "language": "python"},
            timeout=timeout
        )
        res.raise_for_status()
    except requests.RequestException as e:
        return f"--- Sandbox HTTP ERROR ---\n{e}", 1

    res_json = res.json()
    status_ok = res_json.get("status") == "Success"
    run_res = res_json.get("run_result", {})
    stdout = run_res.get("stdout", "")
    stderr = run_res.get("stderr", "")

    if status_ok:
        return stdout or "Execution finished with no stdout.", 0
    else:
        return f"--- Sandbox ERROR ---\n{stderr[-1000:]}", 1

def evaluate_response_vs_answer_with_python_code(response, answer, question_type, rule_id, idx, sandbox_url):
    response_without_code = remove_python_code_snippets(response)
    code_details = {}

    if evaluate_response_vs_answer_without_python_code(response_without_code, answer, question_type, rule_id, idx):
        return True, code_details

    python_code_snippets = extract_python_scripts(response)
    for code in python_code_snippets:
        stdout, return_code = execute_python_code(code, sandbox_url)
        if return_code == 0:
            # Store BOTH code and output in details
            code_details = {
                'executed_code': code,
                'code_output': stdout  # This is critical for LLM verification
            }
            if evaluate_response_vs_answer_without_python_code(stdout, answer, question_type, rule_id, idx):
                return True, code_details
    return False, code_details

def evaluate_response_vs_answer(response, answer, question_type, rule_id, idx, sandbox_url):
    python_code_snippets = extract_python_scripts(response)
    if python_code_snippets:
        is_correct, code_details = evaluate_response_vs_answer_with_python_code(response, answer, question_type, rule_id, idx, sandbox_url)
        return is_correct, code_details
    else:
        return evaluate_response_vs_answer_without_python_code(response, answer, question_type, rule_id, idx), {}

def extract_python_scripts(prediction):
    """
    Extracts all Python code snippets from the prediction text.

    Args:
        prediction (str): The prediction containing Python code.

    Returns:
        List[str]: A list of extracted Python code snippets.
    """
    # Define both types of markers
    start_markers = ["'''python", "```python"]
    end_markers = ["'''", "```"]

    snippets = []

    # Iterate over both types of markers
    for start_marker, end_marker in zip(start_markers, end_markers):
        start_indices = [i for i in range(len(prediction)) if prediction.startswith(start_marker, i)]
        end_indices = [i for i in range(len(prediction)) if prediction.startswith(end_marker, i)]

        for start in start_indices:
            end = next((i for i in end_indices if i > start), None)
            if end is not None:
                # Normal case: both start and end markers found
                snippets.append(prediction[start + len(start_marker):end].strip())
            else:
                # Handle case where start marker exists but no end marker
                # Extract from start marker to end of prediction
                code_candidate = prediction[start + len(start_marker):].strip()
                
                # Try to find where the code likely ends by looking for common patterns
                # that indicate non-code content
                lines = code_candidate.split('\n')
                code_lines = []
                
                for i, line in enumerate(lines):
                    stripped_line = line.strip()
                    
                    # Stop if we encounter patterns that suggest end of code
                    if (stripped_line.startswith('### ') or  # Markdown headers
                        stripped_line.startswith('## ') or
                        stripped_line.startswith('**') or  # Bold markdown
                        stripped_line.startswith('---') or  # Horizontal rule
                        stripped_line.startswith('The ') or  # Natural language explanations
                        stripped_line.startswith('This ') or
                        stripped_line.startswith('Now ') or
                        (stripped_line and not any(c in stripped_line for c in '=()[]{}:.,;') and 
                         len(stripped_line.split()) > 10)):  # Long sentences without code chars
                        break
                    
                    code_lines.append(line)
                
                if code_lines:
                    # Join the code lines and validate it's reasonable Python code
                    extracted_code = '\n'.join(code_lines).strip()
                    if extracted_code:
                        # Try to validate the code is syntactically reasonable
                        try:
                            # Try to compile the code to check for syntax errors
                            # If it fails, progressively remove lines from the end until it works
                            temp_lines = code_lines[:]
                            while temp_lines:
                                try:
                                    test_code = '\n'.join(temp_lines).strip()
                                    if test_code:
                                        compile(test_code, '<string>', 'exec')
                                        extracted_code = test_code
                                        break
                                except (SyntaxError, TypeError):
                                    temp_lines.pop()  # Remove last line and try again
                            
                            if extracted_code:  # Only add if we have valid code
                                snippets.append(extracted_code)
                        except:
                            # If compilation checking fails, just use the original logic
                            if extracted_code:
                                snippets.append(extracted_code)

    return snippets

def compute_one_mixed_question_pass_rate(idx, question_list, response_json):
    if response_json == 'NULL':
        result_dict = {
            "idx": idx,
            "response": response_json,
            "details": None,
            "pass_rate": 0,
            "is_correct": False
        }
        return result_dict
    response_list = extract_all_responses_from_json(response_json)
    correct_num = 0
    results = []
    for q_idx, question in enumerate(question_list):
        category, question_idx = question.rsplit('_', 1)
        question_content = read_json_or_jsonl_with_idx(f'data/{category}', 'sample', idx=question_idx)
        answer = question_content['answer']
        if q_idx >= len(response_list):
            break  
        response = response_list[q_idx]
        response_text = extract_text_from_brackets(response)
        rule_id = question_content['rule_id']
        sandbox_url = "://localhost:8080/run_code"
        is_correct, code_details = evaluate_response_vs_answer(response, answer, category, rule_id, q_idx, sandbox_url)
        if is_correct:
            correct_num += 1  
        results.append({
            "question": question,
            "response_text": response_text,
            "answer": answer,
            "is_correct": is_correct,
            **code_details  # Unpack code details if present
        })
    
    pass_rate = correct_num / len(question_list)
    question_correct = pass_rate == 1.0
    result_dict = {
        "idx": idx,
        "response": response_json,
        "details": results,
        "pass_rate": pass_rate,
        "is_correct": question_correct
    }
    return result_dict

def process_llm_evaluation(result, gold, api_key=None, base_url=None, model_path='gpt-4.1', question_type=None):
    """Helper function to process LLM evaluation in parallel"""
    question_type = result.get('question_type', question_type or 'logic_calculation')
    executed_code = result.get('code_details', {}).get('executed_code', '')
    code_output = result.get('code_details', {}).get('code_output', '')  # Get actual verified output
    
    # Extract sandbox conversation details if available
    sandbox_conversation = result.get('sandbox_conversation', [])
    sandbox_executed_codes = []
    sandbox_outputs = []
    full_conversation = ""
    
    if sandbox_conversation:
        try:
            for message in sandbox_conversation:
                if message is None:
                    continue
                    
                role = message.get("role", "") if isinstance(message, dict) else ""
                content = message.get("content", "") if isinstance(message, dict) else ""
                
                # Build full conversation transcript
                if role == "tool":
                    # For tool messages, format the JSON content nicely
                    full_conversation += f"\n\n{role.upper()}: {content}"
                elif role == "assistant" and "tool_calls" in message:
                    # For assistant messages with tool calls, include both content and tool calls
                    full_conversation += f"\n\n{role.upper()}: {content}"
                    tool_calls = message.get("tool_calls", [])
                    if tool_calls is not None:
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                tool_id = tool_call.get("id", "")
                                function_obj = tool_call.get("function", {})
                                if function_obj is not None:
                                    func_name = function_obj.get("name", "") if isinstance(function_obj, dict) else ""
                                    func_args = function_obj.get("arguments", "") if isinstance(function_obj, dict) else ""
                                    full_conversation += f"\n[TOOL CALL {tool_id}] {func_name}: {func_args}"
                else:
                    full_conversation += f"\n\n{role.upper()}: {content}"
                
                # Extract tool execution results
                if role == "tool":
                    try:
                        import json
                        tool_content = json.loads(content)
                        if tool_content.get("status") == "Success":
                            run_result = tool_content.get("run_result", {})
                            stdout = run_result.get("stdout", "")
                            if stdout:
                                sandbox_outputs.append(stdout)
                    except:
                        pass
                
                # Extract code from assistant messages
                elif role == "assistant":
                    code_blocks = re.findall(r'```python(.*?)```', content, re.DOTALL)
                    for code_block in code_blocks:
                        if code_block.strip():
                            sandbox_executed_codes.append(code_block.strip())
                    
                    # Also check for tool calls with code execution
                    if "tool_calls" in message:
                        tool_calls = message.get("tool_calls", [])
                        if tool_calls is not None:
                            for tool_call in tool_calls:
                                if isinstance(tool_call, dict) and "function" in tool_call:
                                    function_obj = tool_call.get("function", {})
                                    if function_obj is not None and isinstance(function_obj, dict):
                                        func_name = function_obj.get("name", "")
                                        if func_name == "run_python":
                                            args = function_obj.get("arguments", "")
                                            # Parse arguments string as JSON if it's a string
                                            try:
                                                if isinstance(args, str):
                                                    import json
                                                    args_dict = json.loads(args)
                                                    code = args_dict.get("code", "")
                                                elif isinstance(args, dict):
                                                    code = args.get("code", "")
                                                else:
                                                    code = ""
                                                if code and code.strip():
                                                    sandbox_executed_codes.append(code.strip())
                                            except:
                                                pass
        except Exception as e:
            print(f"Error processing sandbox conversation: {str(e)}")
            # Continue with empty values if there's an error
            pass
    
    print(f"Processing LLM evaluation for {question_type} question")
    prompt = ""
    
    # Common code verification section for all prompts

    if question_type == 'graph':
        prompt = f"""You are an expert judge for graph reasoning evaluation. Evaluate whether the model's response is substantively correct, considering:

1. **Number equivalence**:
   - The answer typically contains numbers that represent vertices or edges in a graph.
   - The answer may be formatted differently but should contain the correct numerical values.

2. **Format differences**:
   - Answers may appear as lists, sequences, or natural language descriptions.
   - Focus on whether the correct numerical elements are identified.

**CRITICAL REQUIREMENT**: The correct answer or an equivalent form MUST appear explicitly in the model's response itself. If the correct answer does not appear explicitly in the response, even if you think the reasoning could lead to it or it's highly likely to be derivable, you MUST give a FALSE decision. If the correct answer or equvalent form of correct answer appears, even if the model does bad in following the instructions, e.g. it doesn't present the answer in the specified format ,you MUST give a TRUE decision.

**Evaluation Guidelines**:
- Check if all required numerical values are present explicitly in the response
- Verify that extra values aren't incorrectly included
- The values should correspond to the correct graph elements (vertices, edges, paths, etc.)
- Only judge based on what is explicitly stated, not what could be inferred

Your response MUST use this format:
(AFTER YOUR STEP BY STEP OBSERVATION AND THOUGHTS...)
DECISION: [TRUE/FALSE]
REASON: [Concise explanation of decision]

Now evaluate this case:"""
    elif question_type == 'physics' or question_type == 'phybench':
        prompt = f"""You are an expert judge for physics reasoning evaluation. Evaluate whether the model's response is substantively correct, considering:

1. **Formula equivalence**:
   - Correct: Different but algebraically equivalent forms (e.g. "F = ma" vs "a = F/m")
   - Incorrect: Different formulas that would produce different results (e.g. using kinetic energy formula ½mv² instead of ½mv² + mgh)

2. **Notation and formatting**:
   - Accept: sqrt(3)/3 vs 1/sqrt(3)
   - Accept: A' + BC vs A' + B*C (when multiplication is implied)
   - Reject: Using incorrect symbols (e.g. μ instead of λ for wavelength)

3. **Numerical precision**:
   - Accept: 1.253 vs 1.2529 (within 0.1% ~ difference)
   - Reject: 1.25 vs 1.3 (5% ~ difference without justification)

4. **Term organization**:
   - Accept: Different term ordering if mathematically equivalent
   - Reject: Missing terms or extra terms that change meaning

5. **Code Execution Verification**:
- Prioritize the System Verified Code Output over any claims in the model's response
- If code exists but no System Verified Code Output is shown, the code cannot be considered correct
- Only trust numerical results that appear in both the model's response AND System Verified Code Output

**CRITICAL REQUIREMENT**: The correct answer or an equivalent form MUST appear explicitly in the model's response itself. If the correct answer does not appear explicitly in the response, even if you think the reasoning could lead to it or it's highly likely to be derivable, you MUST give a FALSE decision. If the correct answer or equvalent form of correct answer appears, even if the model does bad in following the instructions, e.g. it doesn't present the answer in the specified format ,you MUST give a TRUE decision.

**Evaluation Guidelines**:
1. Verify algebraic equivalence using symbolic math rules
2. Check all terms are present and properly combined
3. Allow reasonable rounding (3+ decimal places)
4. Penalize incorrect:
   - Physical quantity substitutions
   - Unit conversions
   - Operator precedence errors
   - Vector/scalar mismatches
5. The final answer must be explicitly stated in the model's response. If there is code in the model's response, you should only give the TRUE decision if the output of code is shown and the reference answer is in the code output. If there is no code output, even if the code is correctly written, you should give a FALSE decision.

**Example Evaluation**:
###Model Response###: "[[v² = u² + 2as]]"
###Reference Answer###: "[[v^2 = u^2 + 2*a*s]]"

DECISION: TRUE
REASON: Equivalent notation (² vs ^2, implied multiplication)

Your response MUST use this format:
(AFTER YOUR STEP BY STEP OBSERVATION AND THOUGHTS...)
DECISION: [TRUE/FALSE]
REASON: [Concise explanation of decision]
"""
    elif question_type == 'logic_calculation':
        prompt = f"""You are an expert judge for logic calculation evaluation. Evaluate whether the model's response is substantively correct, considering:

1. **Logical equivalence**:
   - Check if the model's response logically follows from the given conditions.
   
2. **Partial correctness**:
   - If the answer requires multiple values, all required values must be correct.
   - Order may not matter in some cases if the answer represents unordered sets.

3. **Notation and formatting**:
    For example, the model may respond 0 -> 1 -> 0 -> 1, but the answer is 0, 1, 0, 1.
    In this case, you should give a TRUE decision since the model's reasoning is totally correct but only the format is different.

**CRITICAL REQUIREMENT**: The correct answer or an equivalent form MUST appear explicitly in the model's response itself. If the correct answer does not appear explicitly in the response, even if you think the reasoning could lead to it or it's highly likely to be derivable, you MUST give a FALSE decision. If the correct answer or equvalent form of correct answer appears, even if the model does bad in following the instructions, e.g. it doesn't present the answer in the specified format ,you MUST give a TRUE decision.

**Evaluation Guidelines**:
1. Verify if the model's response logically follows from the given conditions.
2. Check if all required values are correct and explicitly stated.
3. The final answer must be explicitly present in the model's response to give a TRUE decision.
4. The correct final answer must appear in that output. If the output is missing or not shown, even if the code is logically correct, mark the decision FALSE.
5. If the correct final answer is clearly stated in natural language without needing to rely on code, you may mark it TRUE.

Your response MUST use this format:
(AFTER YOUR STEP BY STEP OBSERVATION AND THOUGHTS...)
DECISION: [TRUE/FALSE]
REASON: [Concise explanation of decision]
"""
    elif question_type == 'math500' or question_type == 'livemathbench' :
        prompt = f"""You are an expert judge for math reasoning evaluation. Evaluate whether the model's response is substantively correct, considering:

**CRITICAL REQUIREMENT**: The correct answer or an equivalent form MUST appear explicitly in the model's response itself. If the correct answer does not appear explicitly in the response, even if you think the reasoning could lead to it or it's highly likely to be derivable, you MUST give a FALSE decision. If the correct answer or equvalent form of correct answer appears, even if the model does bad in following the instructions, e.g. it doesn't present the answer in the specified format ,you MUST give a TRUE decision.
            
    **Evaluation Guidelines:**
    1. **Mathematical Equivalence**: Determine if the model's answer is mathematically equivalent to the reference answer, even if expressed differently.

    2. **Common Variations to Consider**:
    - Different but equivalent forms (e.g., 2/4 vs 1/2, π²/8 vs π²/8)
    - Different notation styles (e.g., [a,b] vs \\left[a,b\\right])
    - Simplified vs unsimplified expressions
    - Equivalent interval notations
    - Equivalent vector/matrix representations
    - Equivalent trigonometric expressions

    3. **Partial Credit**: For multi-part answers, identify if all parts are correct.

    4. **Formatting Differences**: Ignore differences in LaTeX formatting, brackets, or presentation style if the mathematical content is equivalent.

    5. **Final Answer Extraction**: Look for the final answer which may be preceded by "FINAL ANSWER:" or enclosed in brackets/boxes. The answer must be explicitly present in the response.

    Your response MUST use this format:
    DECISION: [TRUE/FALSE]
    REASON: [Step-by-step mathematical verification explaining why the answers are equivalent or not]
"""
    else:
        prompt = f"""You are an expert judge for logic reasoning evaluation. Evaluate whether the model's response is substantively correct, considering:

1. **Logical equivalence**:
   - Check if the model's response logically follows from the given conditions.
   
2. **Partial correctness**:
    - If the answer requires multiple values, all required values must be correct.
    - Order may not matter in some cases if the answer represents unordered sets.

3. **Notation and formatting**:
    For example, the model may respond 0 -> 1 -> 0 -> 1, but the answer is 0, 1, 0, 1.
    In this case, you should give a TRUE decision since the model's reasoning is totally correct but only the format is different.

**CRITICAL REQUIREMENT**: The correct answer or an equivalent form MUST appear explicitly in the model's response itself. If the correct answer does not appear explicitly in the response, even if you think the reasoning could lead to it or it's highly likely to be derivable, you MUST give a FALSE decision. If the correct answer or equvalent form of correct answer appears, even if the model does bad in following the instructions, e.g. it doesn't present the answer in the specified format ,you MUST give a TRUE decision.

**Evaluation Guidelines**:
1. Verify if the model's response logically follows from the given conditions.
2. Check if all required values are correct and explicitly stated.

Your response MUST use this format:
(AFTER YOUR STEP BY STEP OBSERVATION AND THOUGHTS...)
DECISION: [TRUE/FALSE]
REASON: [Concise explanation of decision]

"""       
    prompt += """IMPORTANT: It is not enough to have correct reasoning method. The model must give the correct answer explicitly in the response to get a TRUE decision. Only judge based on what is explicitly stated in the response, not what could potentially be derived from the reasoning. Now evaluate this case:"""  

    prompt += "\n###Prompt###: " + result.get('prompt', '')
    response_text = result.get('response', '')
    
    # Include full sandbox conversation if available
    if full_conversation:
        prompt += "\n###Full Sandbox Conversation###: " + full_conversation
    elif result.get('full_thinking_response'):
        prompt += "\n###Model Full Response###: " + (str(result.get('full_thinking_response')) or str(result.get('response', '')))
    else:
        # clear the code block in the response text
        response_text = re.sub(r'```python[\s\S]*?```', '', response_text)
        if response_text:
            prompt += "\n###Model Full Response###: " + (response_text or '')
    
    # Add verified execution results to prompt (prioritize sandbox results)
    all_executed_codes = sandbox_executed_codes if sandbox_executed_codes else ([executed_code] if executed_code else [])
    all_outputs = sandbox_outputs if sandbox_outputs else ([code_output] if code_output else [])
    
    if all_executed_codes:
        prompt += f"\n###Executed Code###:\n"
        for i, code in enumerate(all_executed_codes):
            prompt += f"```python\n{code}\n```\n"
            if i < len(all_outputs) and all_outputs[i]:
                prompt += f"Output: {all_outputs[i]}\n"
    
    if all_outputs:
        prompt += f"\n###System Verified Code Output###:\n"
        for output in all_outputs:
            prompt += f"{output}\n"
    else:
        prompt += "\n###System Verified Code Output###: [No output verified by system]"
    
    prompt += "\n\n\n(NOW HERE IS THE REFERENCE ANSWER, NOT THE MODEL'S RESPONSE) ###Reference Answer###: " + gold

    if question_type == 'operation_research':
        pure_gold = extract_text_from_brackets(gold, clean_level='clean')
        # only use isalnum to filter the response text
        pure_response_text = ''.join(char for char in response_text if char.isalnum() or char in [' ', '.', ',', '-', '_', '(', ')'])
        
        # Check both traditional code output and sandbox outputs
        all_code_outputs = all_outputs if all_outputs else ([code_output] if code_output else [])
        found_in_output = False
        for output in all_code_outputs:
            pure_output = ''.join(char for char in output if char.isalnum() or char in [' ', '.', ',', '-', '_', '(', ')'])
            if pure_gold.lower() in pure_output.lower():
                found_in_output = True
                break
        
        if pure_gold.lower() not in pure_response_text.lower() and not found_in_output:
            return False, "The reference answer is not in the model's response, so the LLM judge is ped.", prompt
    try:
        temperature = 0.001
        
        # Use the provided API key and base URL, or fall back to defaults
        # Randomly select between two endpoints to distribute traffic
        # Use time-based seed for better randomization
        random.seed(int(time.time() * 1000000) % 2147483647)
        base_urls = [
            "http://172.30.7.91:23333/v1/",
            "http://172.30.7.92:23333/v1/",
        ]
        selected_base_url = random.choice(base_urls)
        client = OpenAI(base_url=selected_base_url, api_key='YOUR_API_KEY')
        model_name = client.models.list().data[0].id
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        retries = 10
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    timeout=1200
                )
                break  # Success, exit the loop
            except Exception as e:
                if attempt == retries - 1:
                   response = {"choices": [{"message": {"content": "Failed to get response from the LLM Judge after 10 attempts"}}],"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
                # Optionally, add a short delay before retrying
                time.sleep(2**int(attempt))
    except Exception as e:
        error_str = str(e)
        print(f"LLM judge error: {error_str}")
        return False, error_str, prompt
    
    original_response = response.choices[0].message.content
    if response:
        result['LLM_response'] = original_response
        print(f"Successfully got response from the LLM Judge")
        try:
            # Extract decision from response
            if "DECISION:" in original_response:
                extracted_response = original_response.split("DECISION:")[1]
                if "REASON:" in extracted_response:
                    extracted_response = extracted_response.split("REASON:")[0]
                extracted_response = extracted_response.strip()
                result['is_correct'] = "TRUE" in extracted_response
                print(f"result['is_correct']: {result['is_correct']}")

            else:
                print("No DECISION: found in response")
                result['is_correct'] = False
        except Exception as e:
            print(f"Error: {str(e)}, No DECISION: or REASON: in the response")
            result['is_correct'] = False
    # Fallback checks
    if result.get('is_correct') is None:
        result['is_correct'] = False
    
    # Direct match fallback
    answer_text = ""
    if "FINAL ANSWER:" in response_text:
        try:
            final_answer_part = response_text.split("FINAL ANSWER:")[1].strip()
            answer_text = final_answer_part
        except:
            pass
    
    if answer_text and answer_text.strip() == gold.strip():
        result['is_correct'] = True
        
    return result['is_correct'], original_response, prompt

def extract_decision_from_judge_response(judge_response):
    """
    Extract DECISION from an existing judge response.
    
    Args:
        judge_response (str): The existing judge response text
        
    Returns:
        tuple: (is_valid, is_correct) where is_valid indicates if a valid decision was found
    """
    if not judge_response or not isinstance(judge_response, str):
        return False, False
    
    # Patterns to match various DECISION formats
    decision_patterns = [
        r'DECISION:\s*\[?(TRUE|FALSE)\]?',  # DECISION: TRUE, DECISION: [TRUE], etc.
        r'DECISION\s+\[?(TRUE|FALSE)\]?',   # DECISION TRUE, DECISION [FALSE], etc.
    ]
    
    for pattern in decision_patterns:
        match = re.search(pattern, judge_response, re.IGNORECASE)
        if match:
            decision = match.group(1).upper()
            is_correct = decision == 'TRUE'
            print(f"Found existing judge decision: {decision} -> is_correct: {is_correct}")
            return True, is_correct
    
    # If no explicit pattern found, return invalid
    return False, False

def can_reuse_judge_response(result):
    """
    Check if we can reuse an existing judge response for this result.
    
    Args:
        result (dict): Result dictionary that may contain existing judge response
        
    Returns:
        tuple: (can_reuse, is_correct) where can_reuse indicates if we can skip re-evaluation
    """
    # Check for existing judge response in various possible fields
    judge_response_fields = ['judge_response', 'LLM_response', 'llm_response']
    
    for field in judge_response_fields:
        if field in result:
            judge_response = result[field]
            is_valid, is_correct = extract_decision_from_judge_response(judge_response)
            if is_valid:
                print(f"Reusing existing judge response from field '{field}' for result {result.get('idx', 'unknown')}")
                return True, is_correct
    
    # Also check if is_correct is already set and we have a judge response
    if 'is_correct' in result and isinstance(result['is_correct'], bool):
        # If we have is_correct and any judge response field, we can reuse
        for field in judge_response_fields:
            if field in result and result[field]:
                print(f"Reusing existing is_correct value for result {result.get('idx', 'unknown')}")
                return True, result['is_correct']
    
    return False, False

def evaluate_with_llm_judge(results, api_key=None, base_url=None, model_path='Qwen/Qwen2.5-72B-Instruct', max_workers=8,question_type=None):
    """
    Evaluate tasks using an LLM-based judge with parallel processing.
    
    Args:
        results: List of result dictionaries to evaluate
        api_key: API key for the OpenAI/OpenAISDK service
        base_url: Base URL for the API service
        model_path: Path or name of the model to use
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of evaluated results
    """
    if len(results) == 0:
        return []
    question_type = question_type if question_type is not None else 'math500'
    code_details = results[0].get('code_details', {})
    print(f"Starting LLM-based evaluation for {len(results)} {question_type} tasks with {max_workers} workers")
    
    # Create a copy to avoid modifying the original results
    evaluated_results = []
    futures = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in results:
            # Check if we can reuse existing judge response
            can_reuse, is_correct = can_reuse_judge_response(result)
            if can_reuse:
                result['is_correct'] = is_correct
                evaluated_results.append(result)
                continue
                
            # Get the reference answer
            gold = result.get('answer', '')
            if not gold:
                result['is_correct'] = False
                evaluated_results.append(result)
                continue
            
            # Submit task to thread pool
            future = executor.submit(
                process_llm_evaluation,
                result.copy(),  # Pass a copy to avoid concurrent modification
                gold,
                api_key,
                base_url,
                model_path,
                question_type
            )
            futures.append((future, result))

        # Process completed futures
        for i, (future, result) in enumerate(futures):
            try:
                print(f"Processing result {i+1}/{len(futures)}")
                is_correct, judge_response, prompt = future.result(timeout=1200)  # 2 minute timeout
                result['is_correct'] = is_correct
                result['judge_prompt'] = prompt
                result['judge_response'] = judge_response
                evaluated_results.append(result)
                print(f"Completed {i+1}/{len(futures)}: {'✓' if is_correct else '✗'}")
            except TimeoutError:
                print(f"Timeout for result {i+1}/{len(futures)}")
                result['is_correct'] = False
                result['judge_response'] = "TIMEOUT"
                evaluated_results.append(result)
            except Exception as e:
                print(f"Error processing result {i+1}/{len(futures)}: {str(e)}")
                result['is_correct'] = False
                result['judge_response'] = f"ERROR: {str(e)}"
                evaluated_results.append(result)
    
    print(f"Completed LLM-based evaluation: {sum(1 for r in evaluated_results if r.get('is_correct', False))}/{len(evaluated_results)} correct")
    return evaluated_results

def evaluate_responses(data, question_type, mode, use_llm_judge=True, api_key=None, base_url=None, max_workers=8, model_path='Qwen/Qwen2.5-72B-Instruct', tasks_to_judge=['logic_calculation', 'physics', 'phybench', 'math500','aime24','aime25','livemathbench'], sandbox_url="http://localhost:8080/run_code"):
    results = []
    for record in data:
        try:
            idx = record.get("idx")
            response = record.get("response")
            if response is None:
                response = ""  # Initialize as empty string if None
            
            if "sandbox_conversation" in record:
                sandbox_response = ""
                # Convert sandbox_conversation (which is a list of message dicts) to string
                sandbox_content = ""
                for message in record.get("sandbox_conversation", []):
                    role = message.get("role", "")
                    content = message.get("content", "")
                    if "reasoning_content" in message:
                        content += message["reasoning_content"]
                    # Keep code blocks for LLM judge evaluation - don't delete them
                    # content = re.sub(r'```python[\s\S]*?```', '', content)
                    sandbox_content += f"\n\n{role.upper()}: {content}"
                sandbox_response += sandbox_content
                response += sandbox_response
            if "truncated_response" in record:
                response = record.get("truncated_response") + f"FINAL ANSWER: {record.get('new_response')}"
                record["response"] = response
            if tasks_to_judge is not None and question_type not in tasks_to_judge:
                use_llm_judge = False
            if mode == "mixed":
                question_list = record.get("question_list")
                response_json = extract_json(response)
                result_dict = compute_one_mixed_question_pass_rate(idx, question_list, response_json)
                results.append(result_dict)
            else:
                answer = record.get("answer")
                rule_id = record.get("rule_id")
                is_correct, code_details = evaluate_response_vs_answer(response, answer, question_type, rule_id, idx, sandbox_url)
                result_dict = record.copy()
                result_dict["is_correct"] = is_correct
                result_dict["code_details"] = code_details
                if question_type == "counterfactual":
                    real_life_answer = record.get("real_life_answer")
                    is_real_life, _ = evaluate_response_vs_answer(response, real_life_answer, question_type, rule_id, idx, sandbox_url)
                    result_dict["real_life_answer"] = real_life_answer
                    result_dict["is_real_life"] = is_real_life
                if question_type == "cipher" and mode == "subquestions":
                    result_dict["type"] = record.get("type")
                result_dict["response"] = response
                results.append(result_dict)
        except Exception as e:
            print(f"Error processing record {record.get('idx', 'unknown')}: {str(e)}")
            result_dict = record.copy() if isinstance(record, dict) else {"idx": "unknown"}
            result_dict["is_correct"] = False
            result_dict["error_message"] = str(e)
            results.append(result_dict)
    
    
    
    # Original evaluation logic for non-LLM case or non-LLM-judged tasks
    for result_dict in results:
        try:
            # Calculate is_correct based on the submitted answer
            result_dict["answer"] = str(result_dict.get("answer", "")).strip()
            # Extract answer from response if not already present
            try:
                if "new_response" in result_dict:
                    result_dict["response_text"] = str(result_dict["new_response"]).strip()
                elif "response" in result_dict and not result_dict.get("response_text"):
                    response = result_dict["response"]
                    # Handle budget forcing format with FINAL ANSWER: section
                    if "FINAL ANSWER:" in response:
                        final_answer = response.split("FINAL ANSWER:")[1].strip()
                        result_dict["response_text"] = final_answer
                    else:
                        result_dict["response_text"] = response.strip()
            except Exception as e:
                print(f"Error processing response text for result {result_dict.get('idx', 'unknown')}: {str(e)}")
                # Make sure we have a response to use even if processing failed
                if "response" in result_dict:
                    result_dict["response_text"] = str(result_dict["response"]).strip()
                else:
                    result_dict["response_text"] = ""
                    print(f"Warning: No response found for result {result_dict.get('idx', 'unknown')}")
        except Exception as e:
            print(f"Error in post-processing for result {result_dict.get('idx', 'unknown')}: {str(e)}")
            # Ensure basic fields exist even if processing failed
            if "answer" not in result_dict:
                result_dict["answer"] = ""
            if "response_text" not in result_dict:
                result_dict["response_text"] = ""
    if use_llm_judge:
        results = evaluate_with_llm_judge(results, api_key, base_url, max_workers=max_workers, model_path=model_path, question_type=question_type)
    return results

# Add this new function for normalizing math expressions
def normalize_math_expression(expr):
    """Normalize mathematical expressions for better comparison."""
    # Convert to lowercase
    expr = expr.lower()
    
    # Remove spaces, commas and other formatting characters
    expr = re.sub(r'[\s,{}()\[\]]', '', expr)
    
    # Replace \\frac{a}{b} with a/b
    expr = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', expr)
    
    # Replace \\left( and \\right) with simple parentheses
    expr = expr.replace('\\left', '').replace('\\right', '')
    
    # Normalize pi representations
    expr = expr.replace('\\pi', 'pi')
    
    return expr
