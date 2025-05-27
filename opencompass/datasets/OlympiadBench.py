import json
import math
import os
import re
from os import environ
from typing import Dict

import sympy as sp
from datasets import Dataset, DatasetDict
from sympy import Eq, Pow, simplify, sympify
from sympy.parsing.latex import parse_latex

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.registry import (ICL_PROMPT_TEMPLATES, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import get_data_path

from .base import BaseDataset

# Load Dataset


@LOAD_DATASET.register_module()
class OlympiadBenchDataset(BaseDataset):
    """Dataset for OlympiadBench.

    Args:
        path (str): Path to dataset directory
        name (str): Name of specific json file to load
        e.g. 'OE_TO_maths_en_COMP'
    """

    @staticmethod
    def load(path: str, name: str = None, **kwargs):
        """Load dataset.

        Args:
            path (str): Path to dataset directory
            name (str): Name of specific json file to load

        Returns:
            DatasetDict: Dataset with test and train splits
        """
        path = get_data_path(path)
        dataset = DatasetDict()
        raw_data = []

        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset

            ms_dataset = MsDataset.load(path, split='train')
            for item in ms_dataset:
                raw_data.append({
                    'problem':
                    item['question'],
                    'solution':
                    item['final_answer'][0],
                    'language':
                    item['language'],
                    'subject':
                    item['subject'],
                    'question_type':
                    item['question_type'],
                    'answer_type':
                    item['answer_type'],
                    'is_multiple_answer':
                    item['is_multiple_answer'],
                    'unit':
                    item['unit'],
                    'error':
                    item['error'],
                    'questions':
                    item,  # may not be used
                })
        else:
            # Construct file path using name parameter
            if name is None:
                raise ValueError(
                    "Must specify 'name' parameter to load specific json file")

            # file_path = os.path.join(path, name, f'{name}.json')
            file_path = os.path.join(path, f'{name}.json')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f'File not found: {file_path}')

            # Load the specified json file
            data = json.load(open(file_path, encoding='utf-8'))
            for item in data:
                raw_data.append({
                    'problem':
                    item['question'],
                    'solution':
                    item['final_answer'][0],
                    'language':
                    item['language'],
                    'subject':
                    item['subject'],
                    'question_type':
                    item['question_type'],
                    'answer_type':
                    item['answer_type'],
                    'is_multiple_answer':
                    item['is_multiple_answer'],
                    'unit':
                    item['unit'],
                    'error':
                    item['error'],
                    'questions':
                    item,  # may not be used
                })

        dataset['test'] = Dataset.from_list(raw_data)
        dataset['train'] = Dataset.from_list(raw_data)
        return dataset


# Construct Prompt


def get_single_answer_type_text(answer_type, is_chinese):
    if '-' in answer_type:  # No need now
        answer_type = answer_type[:answer_type.find('-')]
    chinese_answer_type_dict = {
        'Numerical': '数值',
        'Expression': '表达式',
        'Equation': '方程',
        'Interval': '区间',
    }
    english_answer_type_dict = {
        'Numerical': 'a numerical value',
        'Expression': 'an expression',
        'Equation': 'an equation',
        'Interval': 'an interval',
    }

    for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
        if t in answer_type:
            if is_chinese:
                return chinese_answer_type_dict[t]
            else:
                return english_answer_type_dict[t]
    raise ValueError(f'Error parsing answer type {answer_type}!')


def get_answer_type_text(answer_type, is_chinese, multiple_answer):
    if ('Need_human_evaluate' in answer_type) or ('Tuple' in answer_type):
        return ''
    if not multiple_answer:
        answer_text = get_single_answer_type_text(answer_type, is_chinese)
        if is_chinese:
            return f'，答案类型为{answer_text}'
        else:
            return (f'The answer of The problem should be '
                    f'{answer_text}. ')
    # Multiple answers case
    if ',' not in answer_type:  # Same answer type for all answers
        answer_text = get_single_answer_type_text(answer_type, is_chinese)
        if is_chinese:
            return f'，题目有多个答案，答案类型均为{answer_text}'
        else:
            return (f'The problem has multiple answers, each of them '
                    f'should be {answer_text}. ')
    # Different answer types
    answer_types = answer_type.split(',')
    answer_types = [
        get_single_answer_type_text(t, is_chinese) for t in answer_types
    ]
    if len(set(answer_types)) == 1:
        answer_text = answer_types[0]
        if is_chinese:
            return f'，题目有多个答案，答案类型均为{answer_text}'
        else:
            return (f'The problem has multiple answers, each of them '
                    f'should be {answer_text}. ')
    else:
        if is_chinese:
            answer_text = '、'.join(answer_types)
            return f'，题目有多个答案，答案类型分别为{answer_text}'
        else:
            answer_text = ', '.join(answer_types)
            return (f'The problem has multiple answers, '
                    f'with the answers in order being {answer_text}. ')


class OlympiadBenchPrompter:

    def __init__(self):
        pass

    def make_prompt(
        self,
        language,
        subject,
        question_type,
        answer_type,
        is_multiple_answer,
        unit,
    ):
        self.is_chinese = language == 'Chinese'
        self.is_math = subject == 'Math'
        self.is_theorem_proving = question_type == 'Theorem proof'
        """Generate prompt based on question properties."""
        if self.is_chinese:
            subject_content = '数学' if self.is_math else '物理'
            if self.is_theorem_proving:
                prompt = (f'以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，'
                          f'运用逻辑推理及常用定理证明题目中的命题。证明过程中使用的变量和公式请使用LaTeX格式表示。')
            else:
                answer_type_text = get_answer_type_text(
                    answer_type,
                    is_chinese=True,
                    multiple_answer=is_multiple_answer,
                )
                if is_multiple_answer:
                    multiple_answer_text = '\\boxed{用英文逗号连接的多个答案}'
                else:
                    multiple_answer_text = '\\boxed{答案}'
                unit_text = ''
                if unit:
                    multiple_answer_text += '(单位)'
                    unit_text = '，注意答案的单位不要放在\\boxed{}中'
                prompt = (f'以下是中国{subject_content}竞赛中的解答题{answer_type_text}。'
                          f'请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的'
                          f'变量和公式请使用LaTeX格式表示。请在最后以"所以最终答案是'
                          f'{multiple_answer_text}。"显式给出结果{unit_text}。')
        else:
            subject_content = 'Math' if self.is_math else 'Physics'
            if self.is_theorem_proving:
                prompt = (
                    f'The following is a theorem proving problem from an '
                    f'International {subject_content} competition. Please use '
                    f'logical reasoning and common theorems to prove the '
                    f'proposition in the problem according to the given '
                    f'requirements. Please use LaTeX format to represent the '
                    f'variables and formulas used in the proof.')
            else:
                if is_multiple_answer:
                    multiple_answer_text = (
                        '\\boxed{multiple answers connected with commas}')
                else:
                    multiple_answer_text = '\\boxed{answer}'
                unit_text = ''
                if unit:
                    multiple_answer_text += '(unit)'
                    unit_text = (', note that the unit of the answer should '
                                 'not be included in \\boxed{}')
                answer_type_text = get_answer_type_text(
                    answer_type,
                    is_chinese=False,
                    multiple_answer=is_multiple_answer,
                )
                prompt = (
                    f'The following is an open-ended problem from an '
                    f'International {subject_content} competition. '
                    f'{answer_type_text}Please calculate the answer according '
                    f'to the given requirements and the information provided. '
                    f'Please use LaTeX format to represent the variables and '
                    f'formulas used in the solution process and results. '
                    f'Please end your solution with "So the final answer is '
                    f'{multiple_answer_text}." and give the result explicitly'
                    f'{unit_text}.')
        # Add problem statement to the prompt
        prompt = prompt + '\n' + '{problem}' + '\n'
        # Add step-by-step reasoning instruction
        if self.is_chinese:
            prompt += '\n请通过逐步推理来解答问题，并把最终答案放置于\\boxed{}中。'
        else:
            prompt += ('\nPlease reason step by step, and put your final '
                       'answer within \\boxed{}.')
        return prompt


# Evaluate


class MathJudger:

    def __init__(self):
        self.special_signal_map = {
            '\\left': '',
            '\\right': '',
            '∶': ':',
            '，': ',',
            '$': '',
            '\\approx': '=',
            '\\simeq': '=',
            '\\sim': '=',
            '^\\prime': "'",
            '^{\\prime}': "'",
            '^\\circ': '',
            '%': '',
        }
        self.pi = parse_latex('\\pi')
        self.precision = 1e-8

    def split_by_comma(self, expr: str):
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char == '(' or char == '[':
                in_bracket_num += 1
            elif char == ')' or char == ']':
                in_bracket_num -= 1
            elif char == ',' and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())

        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list):
        new_expr_list = []
        for expr in expr_list:
            if '\\pm' in expr:
                new_expr_list.append(expr.replace('\\pm', '+'))
                new_expr_list.append(expr.replace('\\pm', '-'))
            else:
                new_expr_list.append(expr)

        return new_expr_list

    def judge(self, expression1, expression2, precision=1e-8):
        # (默认 expression1 为 Ground_Truth)
        precision = precision if type(precision) == list else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1,
                                                       expression2)
        except Exception:  # 处理具体异常
            return False
        if expression1 == expression2:
            return True

        # 去除字符串中的中文字符
        expression1 = re.sub(r'[\u4e00-\u9fff]+', '', expression1)
        expression2 = re.sub(r'[\u4e00-\u9fff]+', '', expression2)

        expression1 = self.split_by_comma(expression1)
        expression2 = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1)
        temp_list2 = self.trans_plus_minus_sign(expression2)

        # 设计误差值列表
        if len(precision) <= 1:
            precision = precision * len(temp_list1)

        if len(temp_list1) != len(temp_list2):
            return False

        # 判断两个列表中的元素是否可以两两配对，并且两两相等
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]

            for item2 in temp_list2:
                if self.is_equal(item1, item2):
                    temp_list1.remove(item1)
                    temp_list2.remove(item2)
                    precision.remove(self.precision)
                    break
            else:
                return False

        # 如果所有元素都匹配并移除，列表可以配对
        return True

    def is_interval(self, epr):
        return epr.startswith(('(', '[')) and epr.endswith((')', ']'))

    def sympy_sub_pi(self, expression_sympy):
        return expression_sympy.subs(self.pi, math.pi)

    def is_equal(self, expression1, expression2):
        if (expression1 == expression2 and expression1 != ''
                and expression2 != ''):
            return True

        # 先判断是否是两个区间
        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2):
                    return True
            except Exception:  # 处理具体异常
                return False

        # 再判断是否在数值上相等
        try:
            if self.numerical_equal(expression1, expression2):
                return True
        except Exception:  # 处理具体异常
            pass

        # 再判断是否是表达式相等
        try:
            if self.expression_equal(
                    expression1, expression2) and not ('=' in expression1
                                                       and '=' in expression2):
                return True
        except Exception:  # 处理具体异常
            pass

        # 再判断是否是等式相等
        try:
            if self.equation_equal(expression1, expression2):
                return True
        except Exception:  # 处理具体异常
            pass

        return False

    def numerical_equal(
        self,
        expression1: str,
        expression2: str,
        include_percentage: bool = True,
    ):
        """(默认 expression1 为 Ground_Truth) 函数: 判读两个数值是否在误差允许范围内相等 步骤1:

        将可能出现的百分号的情况包含进来 步骤2: 使用 math.isclose 函数判断是否相等.
        """
        reference = float(expression1)
        prediction = float(expression2)

        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]

        for item in gt_result:
            if abs(item - prediction) <= self.precision * 1.01:
                return True
        return False

    def expression_equal(self, exp1, exp2):
        """(默认 expression1 为 Ground_Truth) 函数: 判断两个表达式是否在数学意义上等价 步骤1: 提取表达式,
        防止有的模型会给出"x=1"而不是"1" 步骤2: 使用 sympy 库进行等价判断."""

        # 只提取等号右边的表达式
        def extract_expression(expression):
            if '=' in expression:
                expression = expression.split('=')[1]
            return expression.strip()

        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        # 将表达式转换为 sympy 中能够进行处理的格式
        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))

        if expr1_sym == expr2_sym:
            return True
        else:
            expr1_sym = self.sympy_sub_pi(expr1_sym)
            expr2_sym = self.sympy_sub_pi(expr2_sym)

            if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (
                    not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
                return False
            elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    if not (self.can_compute_power(expr1_sym)
                            and self.can_compute_power(expr2_sym)):
                        print(f'These two number can not be calculated by '
                              f'current computer for: '
                              f'"{str(expr1_sym)}" and "{str(expr2_sym)}"')
                        return False

                    if (abs(expr1_sym.evalf() - expr2_sym.evalf()) <=
                            self.precision * 1.01):
                        return True
                    else:
                        return False
                except Exception:  # 处理具体异常
                    return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)

                    num_value = simplified_expr.evalf()

                    return abs(num_value) < 1e-3
                except Exception:  # 处理具体异常
                    return False

    def equation_equal(self, expression1, expression2):
        """
        (expression1 is assumed to be Ground_Truth)
        Function: Check if two equations are mathematically equivalent
        Step 1: Simplify equations to standard form with right side equal to 0
        Step 2: Use sympy library to calculate quotient of left sides,
        if quotient or its reciprocal is integer, equations are equivalent
        """

        # Convert equations to sympy format with right side moved to left side
        def simplify_equation(latex_eq):
            # Split left and right sides of equation
            lhs, rhs = latex_eq.split('=')

            # Parse LaTeX expressions using parse_latex
            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)

            # Create equation object
            equation = Eq(lhs_expr, rhs_expr)

            # Simplify equation by moving right side to left
            simplified_eq = simplify(equation.lhs - equation.rhs)

            return simplified_eq

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)

        # If division result or its reciprocal is
        # non-zero integer, equations are equivalent
        if (division_result_1.is_Integer
                and division_result_1 != 0) or (division_result_2.is_Integer
                                                and division_result_2 != 0):
            return True
        else:
            return False

    def interval_equal(self, expression1, expression2):
        """
        Function: Check if two intervals are mathematically equivalent
        Step 1: Simplify interval expressions,
                remove irrelevant symbols
                like "\\left", "\\right", and "x \\in"
        Step 2: Compare brackets and mathematical expressions in between
        """

        def compare_two_interval(inter1, inter2):
            # First compare brackets on both sides
            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False

            inter1 = inter1.strip('[]()')
            inter2 = inter2.strip('[]()')

            # Split interval into left and right parts
            items_1 = inter1.split(',')
            items_2 = inter2.split(',')

            for item_1, item_2 in zip(items_1, items_2):
                if not self.expression_equal(item_1, item_2):
                    return False
            return True

        interval1 = expression1
        interval2 = expression2

        if interval1 == interval2:
            return True
        else:
            inter_list1 = interval1.split('\\cup')
            inter_list2 = interval2.split('\\cup')

            if len(inter_list1) != len(inter_list2):
                return False
            else:
                for inter1, inter2 in zip(inter_list1, inter_list2):
                    if not compare_two_interval(inter1, inter2):
                        return False
                return True

    def preprocess(self, expression1, expression2):
        """Extract and preprocess expressions from model output."""

        def extract_boxed_content(latex_str):
            # Find all \boxed{...} structures
            boxed_matches = re.finditer(r'\\boxed{', latex_str)
            results = ''

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                # Search from after \boxed{ until
                # finding matching closing brace
                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == '{':
                        stack += 1
                    elif latex_str[end_index] == '}':
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    # Extract content inside \boxed{}
                    content = latex_str[start_index:end_index - 1]
                    results += content + ','
                else:
                    raise ValueError('Mismatched braces in LaTeX string.')

            # If no \boxed{} found, extract formulas from last line
            if results == '':
                last_line_ans = latex_str.strip().split('\n')[-1]
                dollar_pattern = r'\$(.*?)\$'
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ','
                else:
                    results = latex_str

            return results

        def special_symbol_replace(expression):
            if '\\in ' in expression:
                expression = expression.split('\\in ')[1]

            # Replace special characters that
            # don't affect LaTeX parsing (decorative)
            for signal in self.special_signal_map:
                expression = expression.replace(
                    signal, self.special_signal_map[signal])

            expression = expression.strip('\n$,.:;^_=+`!@#$%^&*~，。')

            pattern = r'\\(?:mathrm|mathbf)\{~?([^}]*)\}'
            expression = re.sub(pattern, r'\1', expression)

            return expression

        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(
            expression2)
        exp1, exp2 = special_symbol_replace(exp1), special_symbol_replace(exp2)

        return exp1, exp2

    def can_compute_power(self, expr):
        """Check if the power expression can be computed.

        Parameters:
        expr (sympy expression): The expression to check.

        Returns:
        bool: True if the expression can be computed, False otherwise.
        """
        # Check if the expression is a power expression
        if isinstance(expr, Pow):
            # Extract the base and the exponent
            base, exp = expr.as_base_exp()

            # Check if the base and the exponent are numbers
            if base.is_number and exp.is_number:
                # Set a threshold for the maximum size of the exponent
                # can be adjusted based on the computing environment
                MAX_EXP = 1000

                # Check if the exponent is greater than the threshold
                if abs(exp.evalf()) > MAX_EXP:
                    return False
                else:
                    return True
            else:
                # If the base or the exponent is not a number,
                # we cannot compute the power
                return False
        else:
            # If the expression is not a power expression,
            # return True as it is not the case we are checking for
            return True


@TEXT_POSTPROCESSORS.register_module('olympiadbench_postprocess_v2')
def olympiadbench_postprocess_v2(text: str,
                                 is_chinese: bool = False,
                                 is_deepseek: bool = False) -> str:
    """Extract answer from model output."""
    # deepseekmath has special answering format
    if is_deepseek:
        if is_chinese:
            matches = re.findall('## 解题答案(.*)', text)
        else:
            matches = re.findall('The answer is: (.*)', text)
    else:
        if is_chinese:
            matches = re.findall('所以最终答案是(.*)', text)
        else:
            matches = re.findall('So the final answer is (.*)', text)

    # If found matches, take the last one, otherwise return the whole text
    if matches:
        return matches[-1].strip()
    return text


class OlympiadBenchEvaluator(BaseEvaluator):
    """Evaluator for OlympiadBench dataset."""

    def __init__(self, version='v1'):
        assert version in ['v1', 'v2']
        self.version = version
        self.judger = MathJudger()

    def score(self, predictions, references):  # Remove questions parameter
        """Calculate accuracy score.

        Args:
            predictions (list): List of model predictions
            references (list): List of ground truth answers
        """
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        correct = 0
        count = 0
        details = []

        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            count += 1

            # Get precision/error threshold from reference if available
            precision = 1e-8
            if isinstance(ref, dict) and 'error' in ref:
                if ',' in ref['error']:
                    # Multiple precisions for multiple answers
                    precisions = ref['error'].split(',')
                    precisions = [float(p) if p else 1e-8 for p in precisions]
                    precision = precisions
                else:
                    precision = float(ref['error'])

            # Check if answer is correct
            try:
                if (isinstance(ref, dict) and 'answer_type' in ref
                        and 'Tuple' in ref['answer_type']):
                    # Special handling for tuple type answers
                    is_correct = self.judger.judge(pred,
                                                   ref['final_answer'][0],
                                                   precision)
                else:
                    is_correct = self.judger.judge(pred, ref, precision)

                if is_correct:
                    correct += 1
                    detail['correct'] = True
            except Exception as e:  # 处理具体异常
                detail['error'] = str(e)

            details.append(detail)

        result = {'accuracy': 100 * correct / count, 'details': details}
        return result


@ICL_PROMPT_TEMPLATES.register_module()
class OlympiadBenchTemplate(PromptTemplate):
    """Template for OlympiadBench dataset."""

    def __init__(self):
        # Define basic template structure
        template = dict(round=[dict(role='HUMAN', prompt='{prompt}')])
        super().__init__(template=template)
        self.prompter = OlympiadBenchPrompter()

    def generate_item(self, entry: Dict, *args, **kwargs) -> str:
        """Generate prompt for a single item."""
        problem = entry.get('problem', '')
        language = entry.get('language', 'English')
        subject = entry.get('subject', 'Math')
        question_type = entry.get('question_type', '')
        answer_type = entry.get('answer_type', '')
        is_multiple_answer = entry.get('is_multiple_answer', False)
        unit = entry.get('unit', '')

        prompt = self.prompter.make_prompt(
            language=language,
            subject=subject,
            question_type=question_type,
            answer_type=answer_type,
            is_multiple_answer=is_multiple_answer,
            unit=unit,
        )

        new_entry = {'prompt': prompt, 'problem': problem}

        return super().generate_item(new_entry, *args, **kwargs)
