import os
import re
import pandas as pd
import numpy as np
import requests
import json
import sys
import sympy as sp
from sklearn.metrics import r2_score,root_mean_squared_error
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MLLM_claudeshop = {
    'gpt-3.5': 'gpt-3.5-turbo',
    'gpt-4o': 'chatgpt-4o-latest',
    'gpt-4': 'gpt-4',
    'gpt-o3': 'o3-mini',
    'claude-3-7': 'claude-3-7-sonnet-20250219-thinking',
    'Qwen-72b': 'qwen-72b',
    'Qwen2.5':'qwen2.5-32b-instruct',
    'Qwen-vl': 'qwen-vl-max',
    'Gemini-1.5p': 'gemini-1.5-pro-latest',
    'Gemini-2.0p': 'gemini-2.0-pro-exp-02-05',
    'Gemini-2.5p': 'gemini-2.5-pro-exp-03-25',
    'grok-2': 'grok-2',
    'grok-3': 'grok-3',
}

MLLM_siliconflow = {
    'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
    'deepseek-r1': 'Pro/deepseek-ai/DeepSeek-R1',
    'QwQ-32b': 'Qwen/QwQ-32B',
    'Qwen2.5-vl-72b': 'Qwen/Qwen2.5-VL-72B-Instruct',
}

MLLM_intern = {
    'InternLM3-8B': 'internlm3-8b-instruct',
    'InternVL3-78B': 'internvl2.5-78b',
}

MLLM_other = {
    'MOE': 'MOE',
}

def _send_request(messages, mllm='4o'):
    
    if mllm in MLLM_claudeshop:
        URL = f"your_url_here"  # Replace with the actual URL
        API_KEY = "your_api_key_here"  # Replace with the actual API key
        HEADERS = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {API_KEY}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        model = MLLM_claudeshop[mllm]
    elif mllm in MLLM_siliconflow:
        URL = f"your_url_here"  # Replace with the actual URL
        API_KEY = "your_api_key_here"  # Replace with the actual API key
        HEADERS = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        model = MLLM_siliconflow[mllm]
    elif mllm in MLLM_intern:
        URL = f"your_url_here"  # Replace with the actual URL
        API_KEY = "your_api_key_here"  # Replace with the actual API key
        HEADERS = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        model = MLLM_intern[mllm]
    elif mllm in MLLM_other:
        URL = f"your_url_here"  # Replace with the actual URL
        API_KEY = "your_api_key_here"  # Replace with the actual API key
        HEADERS = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        model = MLLM_other[mllm]
        
        
    count = 0
    while True and count < 20:
        count += 1
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 50
        })
        session = requests.Session()
        session.keep_alive = False
        response = session.post(URL, headers=HEADERS, data=payload, verify=True)
        try:
            content = response.json()['choices'][0]['message']['content']
            break
        except:
            content=None
            pass
    
    return content
    
        


def llm_formula(formula, var_list, mllm='gpt-4o'):
    content = f'''
        You are provided with a mathematical formula involving multiple variables. Your task is to rewrite this formula in the form of y=f(x0,x1,...).
        The formula is as follows:
        {formula}
        The variables in the formula are denoted as: {', '.join(var_list)}.
        Replace them in the order they appear with x0, x1, x2, ..., and replace the dependent variable with y.
        Please output only the reformulated equation, in the form y=x0,x1,..., without any additional information.
    '''
    messages = [{"role": "user", "content": content}]
    content = _send_request(messages, mllm=mllm)
    return content


def clean_formula_string(formula_str):
    # 1. 删除 Markdown 残留符号
    formula_str = formula_str.replace('×', '*').replace('·', '*').replace('÷', '/')
    formula_str = formula_str.replace('−', '-').replace('^', '**')
    formula_str = formula_str.replace('“', '"').replace('”', '"').replace('’', "'")

    # 2. 去除 markdown 反引号 ``` 和 $ 符号
    formula_str = formula_str.replace('`', '').replace('$', '').strip()

    # 3. 提取第一行公式（防止有多行解释性输出）
    formula_str = formula_str.split('\n')[0].strip()

    # 4. 用正则去除非合法字符（保留基本数学表达式）
    formula_str = re.sub(r'[^\w\s\+\-\*/\^\=\.\(\)]', '', formula_str)

    # 5. 确保左右去空格
    return formula_str.strip()

def llm_evaluate(inferred_formula, true_formula, mllm='gpt-4o'):
    content = f'''
        You are given two mathematical formulas. Your task is to evaluate how structurally similar they are, and return a similarity score between 0 and 1.

        The score should reflect how closely the formulas match in terms of:
        - Mathematical operations and structure (e.g., same use of +, *, sin, etc.)
        - Term arrangement and complexity
        - Overall symbolic expression and intent

        A score of:
        - 1 means the formulas are structurally identical or mathematically equivalent
        - Around 0.8-0.9 means they are very similar but not identical
        - Around 0.5 means moderately similar (e.g., same overall shape but different terms)
        - Near 0 means structurally unrelated formulas

        Do not consider numerical evaluation or specific input values — only the symbolic structure and mathematical form.

        Formulas:
        Inferred Formula: {inferred_formula}
        True Formula: {true_formula}

        ONLY RETURN [THE SIMILARITY SCORE]
    '''
    messages = [{"role": "user", "content": content}]
    similarity_score = _send_request(messages, mllm=mllm)
    return similarity_score[-4:]

def llm_translate(dirty_formula, mllm='gpt-4o'):
    content = f'''
        This is a language model's judgment on a mathematical formula. Please help me extract the mathematical formula from this judgment and return it:
        {dirty_formula}
        Please serve pi as pi and use x0, x1, x2,... to represent the variable names.
        ONLY RETURN THE FORMULA STRING (Not LATEX).
    '''
    messages = [{"role": "user", "content": content}]
    clean_formula = _send_request(messages, mllm=mllm)
    return clean_formula

def is_symbolically_equivalent(formula1, formula2, n_var=2):
    try:
        x = [sp.Symbol(f'x{i}') for i in range(n_var)]

        expr1 = sp.sympify(formula1.split('=')[1] if '=' in formula1 else formula1)
        expr2 = sp.sympify(formula2.split('=')[1] if '=' in formula2 else formula2)

        return sp.simplify(expr1 - expr2) == 0
    except Exception:
        return False

def parse_formula(formula_str, n_var=2):
    try:
        if '=' in formula_str:
            _, expr_str = formula_str.split('=', 1)
        else:
            expr_str = formula_str
        variables = [sp.Symbol(f'x{i}') for i in range(n_var)]
        expr = sp.sympify(expr_str)
        func = sp.lambdify(variables, expr, modules='numpy')
        return func
    except Exception as e:
        print(f'[Parse Error] {formula_str}\n{e}')
        return None

def evaluate_formula_metrics(formula_str, true_formula, x, y_true, n_var=2, mllm='gpt-4o'):
    metrics = {
        'LLM_Score': None,
        'RMSE': None,
        'SymbolicMatch': False,
        'R2': -100000.0
    }

    # 结构评分（用 LLM）
    metrics['LLM_Score'] = llm_evaluate(formula_str, true_formula, mllm=mllm)

    # 数值拟合
    func = parse_formula(formula_str, n_var)
    if func is not None:
        try:
            x_vars = [x[:, i] for i in range(n_var)]
            y_pred = func(*x_vars)
            if np.isscalar(y_pred):
                y_pred = np.full_like(y_true, y_pred)
            metrics['RMSE'] = root_mean_squared_error(y_true, y_pred)
            metrics['R2'] = r2_score(y_true, y_pred)
        except Exception:
            pass

    # 判断方程等价性
    metrics['SymbolicMatch'] = is_symbolically_equivalent(formula_str, true_formula, n_var)

    return metrics



mllm = 'gpt-4o'
sample_num = 100
n_var = 2

os.makedirs(f'{n_var}d/', exist_ok=True)
for seed_idx in [1]:
    try:
        formula_2d = pd.read_csv(f'{n_var}d/Feynman_{n_var}d.csv')
    except:
        formula_2d = pd.DataFrame(columns=['Formula', 'Filename', 'n_variables'])

        collect = pd.read_csv('Feynman/FeynmanEquations.csv')
        try:
            for index, row in collect.iterrows():
                file_path = f'Feynman/Feynman_with_units/' + str(row['Filename'])
                formula = row['Formula']
                n_variables = int(row['# variables'])
                
                if n_variables == n_var:
                    try:
                        dataset = np.loadtxt(file_path)
                    except:
                        continue
                    if dataset.shape[1] == n_variables + 1:
                        var_list = [row[f'v{var_idx+1}_name'] for var_idx in range(n_variables)]
                        new_formula = llm_formula(formula, var_list)
                        print(index, formula, '——>', new_formula)
                    else:
                        continue
                    formula_2d = formula_2d._append({'Formula': new_formula, 'Filename': row['Filename'], 'n_variables': n_variables}, ignore_index=True)
        except Exception as e:
            print(e)

        formula_2d.to_csv(f'{n_var}d/Feynman_{n_var}d.csv', index=False)
    
    try:
        result = pd.read_csv(f'{n_var}d/Feynman_{n_var}d_s{sample_num}_{mllm}.csv')
    except:
        result = pd.DataFrame({
            'Index': pd.Series(dtype=int),
            'GT': pd.Series(dtype=str),
            'Pred': pd.Series(dtype=str),
            'Score': pd.Series(dtype=float),
            'RMSE': pd.Series(dtype=float),
            'R2': pd.Series(dtype=float),
            'SymbolicMatch': pd.Series(dtype=bool)
        })

    for index, row in formula_2d.iterrows():
        true_formula = row['Formula']
        file_path = f'Feynman/Feynman_with_units/' + str(row['Filename'])
        dataset = np.loadtxt(file_path)
        rand_idx = np.random.choice(dataset.shape[0], sample_num, replace=False)
        dataset = dataset[rand_idx]
        x = dataset[:, :n_var]
        y_true = dataset[:, -1]
        
        data_samples = '\n'.join([f'x0={x1:.4f}, x1={x2:.4f}, y={y:.4f}' for x1, x2, y in dataset[:-1]])
        content = f'''
            You will be provided with a set of input-output pairs. Based on these data, infer the mathematical relationship between y and multiple input variables. Please note that the possible mathematical operations include: +, -, *, /, exp, sqrt, sin, arcsin, and constant terms.
            The input sample data are as follows:
            {data_samples}
            Based on the above data, please infer the possible formula. Ensure that your inference applies to all the provided data points, and consider both linear and nonlinear combinations.
            Verify whether your formula applies to the following new data point and adjust it to ensure accuracy:
            {f'x0={dataset[-1, 0]:.4f}, x1={dataset[-1, 1]:.4f}, y={dataset[-1, 2]:.4f}'}
            Finally, please output only the formula string you inferred (e.g. z=x_0 * x_1), without any additional information.
        '''
        messages = [{"role": "user", "content": content}]

        infer_formula  = _send_request(messages, mllm=mllm)
        infer_formula  = llm_translate(infer_formula, mllm='gpt-4o') 
        infer_formula  = clean_formula_string(infer_formula)
        metrics = evaluate_formula_metrics(infer_formula, true_formula, x, y_true, n_var=n_var, mllm='gpt-4o')
        
        print(f'GT: {true_formula.ljust(40)} | Pred: {infer_formula.ljust(40)} | Score: {metrics["LLM_Score"]} | RMSE: {metrics["RMSE"]} | R2: {metrics["R2"]} | Match: {metrics["SymbolicMatch"]}')
        result = result._append({
            'Index': seed_idx,
            'GT': true_formula,
            'Pred': infer_formula,
            'Score': metrics['LLM_Score'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2'],
            'SymbolicMatch': bool(metrics['SymbolicMatch'])
        }, ignore_index=True)

    result.to_csv(f'{n_var}d/Feynman_{n_var}d_s{sample_num}_{mllm}.csv', index=False)
    if not result.empty:
        symbolic_accuracy = result['SymbolicMatch'].sum() / len(result)
        print(f'\model: {mllm},sample_nums: {sample_num},symbolic_accuracy: {symbolic_accuracy:.4f}')
    else:
        symbolic_accuracy = 0
    csv_filepath = f'{n_var}d/Feynman_{n_var}d_s{sample_num}_{mllm}.csv'
    result.to_csv(csv_filepath, index=False)

    with open(csv_filepath, 'a', encoding='utf-8') as f:
        f.write("symbolic_accuracy:"+f'{symbolic_accuracy:.4f}')
        f.write(f"AverageR2,{average_r2:.4f}\n")

    