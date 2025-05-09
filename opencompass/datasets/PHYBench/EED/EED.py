import re

import timeout_decorator
from latex2sympy2_extended import latex2sympy
from sympy import (Add, Float, Function, Integer, Mul, Pow, Rational, Symbol,
                   expand, simplify)
from sympy.core.numbers import Exp1, Infinity, NegativeInfinity, Pi
from sympy.simplify import posify

from opencompass.datasets.PHYBench.EED.extended_zss import ext_distance


def brackets_balanced(s: str) -> bool:
    stack = []
    bracket_pairs = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in bracket_pairs.values():
            stack.append(char)
        elif char in bracket_pairs:
            if not stack or stack[-1] != bracket_pairs[char]:
                return False
            stack.pop()
    return len(stack) == 0


def remove_non_ascii(text):
    return text.encode('ascii', errors='ignore').decode()


def extract_bracket_content(s: str, bracket_position: int) -> str:
    start_idx = bracket_position

    content = []
    escaped = False
    brace_start = start_idx + 1
    brace_depth = 0
    for i in range(brace_start, len(s)):
        char = s[i]
        if escaped:
            content.append(char)
            escaped = False
            continue
        if char == '\\':
            escaped = True
            content.append(char)
            continue
        if char == '{':
            brace_depth += 1
            content.append(char)
        elif char == '}':
            if brace_depth == 0:
                return ''.join(content), i
            brace_depth -= 1
            content.append(char)
        else:
            content.append(char)

    return None, -1


def find_first_unescaped_brace(s: str) -> int:
    escaped = False
    for i, c in enumerate(s):
        if c == '\\' and not escaped:
            escaped = True
            continue
        if c == '{' and not escaped:
            return i
        escaped = False
    return -1


def extract_command(s: str, brace_pos: int) -> str | None:
    """extract the command name from a bracket."""
    i = brace_pos - 1
    parameter_mode = False
    while i >= 0:
        if not parameter_mode and s[i] in ('^', '_'):
            return s[i]
        if not parameter_mode and not s[i] in (' ', '\t', ']', '['):
            break
        if s[i] == ']':
            parameter_mode = True
        if s[i] == '[' and parameter_mode:
            parameter_mode = False
        i -= 1

    # Start point
    if i < 0 or s[i] == '\\':
        return None

    # Extract command name
    command_end = i
    i -= 1
    while i >= 0 and s[i].isalpha():
        i -= 1
    if i < -1 or s[i] != '\\':
        return None
    return s[i + 1:command_end + 1]


def remove_command(s, command, keep_inside=False):
    pos = s.find(command)
    if pos < 0:
        return s
    end_index = pos + len(command)
    level = 0
    if end_index < len(s) and s[end_index] == '{':
        while end_index < len(s):
            if s[end_index] == '{':
                level += 1
            elif s[end_index] == '}':
                level -= 1
                if level == 0:
                    break
            end_index += 1
    else:
        s1 = ''.join([s[0:pos], s[end_index:]])
    if keep_inside:
        s1 = ''.join(
            [s[0:pos], s[pos + len(command) + 1:end_index], s[end_index + 1:]])
    else:
        s1 = ''.join([s[0:pos], s[end_index + 1:]])

    if command not in s1:
        return s1
    else:
        return remove_command(s1, command, keep_inside)


def convert_latex_fractions(latex_str):
    """Convert non-standard fraction like \frac\alpha2 to its standard-
    convertable \frac{\alpha}{2}.

    We support single letter, number or standard form.
    """
    pattern = (r'\\frac((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))'
               r'((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))')

    def replacer(match):
        numerator, denominator = match.group(1), match.group(2)
        wrap_num = f'{{{numerator}}}' if not (
            numerator.startswith('{')
            and numerator.endswith('}')) else numerator
        wrap_den = f'{{{denominator}}}' if not (
            denominator.startswith('{')
            and denominator.endswith('}')) else denominator
        return fr'\frac{wrap_num}{wrap_den}'

    return re.sub(pattern, replacer, latex_str)


def get_first_brace_command(s: str) -> str | None:
    """Find the first brace."""
    brace_pos = find_first_unescaped_brace(s)
    if brace_pos == -1:
        return None
    return extract_command(s, brace_pos)


def remove_overall_brace(s: str) -> str:
    """Remove the overall {xxx} brace."""
    pos = find_first_unescaped_brace(s)
    if pos == -1:
        return s, 0
    command = get_first_brace_command(s)
    if not command:
        content, final = extract_bracket_content(s, pos)
        if final == len(s) or '}' not in s[final + 1:]:
            return content, 1
    return s, 0


def exp_frac(s):

    def exp_frac_single(s):
        position = s.find('^\\frac') + 1
        if position == 0:
            return s
        level = 0
        cnt = 0
        idx = position
        while idx < len(s):
            if s[idx] == '{':
                cnt += 1
            elif s[idx] == '}':
                cnt -= 1
                if cnt == 0:
                    level += 1
                    if level == 2:
                        break
            idx += 1
        s1 = ''.join([s[0:position], '{', s[position:idx], '}', s[idx:]])
        return s1

    s1 = exp_frac_single(s)
    cnt = 0
    while s1 != s and cnt < 100:
        cnt += 1
        s = s1
        s1 = exp_frac_single(s)
    return s


def find_all(s, sub_str, allow_overlap=True):
    indexes = []
    start = 0
    step = 1 if allow_overlap else len(sub_str)
    cnt = 0
    while True and cnt < 100:
        pos = s.find(sub_str, start)
        if pos == -1:
            break
        indexes.append(pos)
        start = pos + step
        cnt += 1
    return indexes


def bar_inside_vec(s):
    indices = find_all(s, '\\vec{')
    if not indices:
        return s
    for i in range(len(indices)):
        position = find_all(s, '\\vec{')[i]
        idx = position + 4
        idx2 = idx
        level = 0
        while idx2 < len(s):
            if s[idx2] == '{':
                level += 1
            if s[idx2] == '}':
                level -= 1
                if level == 0:
                    break
            idx2 += 1

        s1 = s[idx + 1:idx2]

        s1 = remove_command(s1, '\\bar', keep_inside=True)
        s2 = ''.join([s[0:idx + 1], s1, s[idx2:]])
        s = s2
    return s


def vec_lower_idx(input_str):
    pattern = r'\\vec\{([^{}]+)_{([^{}]+)}\}'
    replacement = r'\\vec{\1}_{\2}'
    return re.sub(pattern, replacement, input_str)


def convert_vec_syntax(text):
    pattern = r'\\vec(\s*)(\\?[a-zA-Zα-ωΑ-Ω]+)'
    replacement = r'\\vec{\2}'
    return re.sub(pattern, replacement, text)


def remove_outer_braces(tex_str):
    pattern = r'\{(\\(?:[a-zA-Z]+|.)|[^{}])+\}_\{([^}]+)\}'
    return re.sub(pattern, r'\1_{\2}', tex_str)


def extract_last_equal_content(s: str, strip_whitespace: bool = True) -> str:

    comparison_operators = ('=', '\\approx', '\\ge', '\\le', '\\geq', '\\leq',
                            '<', '>')

    content = s
    for sign in comparison_operators:
        if sign in s:
            rfind_index = s.rfind(sign)
            if rfind_index != -1:
                content = s[rfind_index + 1:]
    if strip_whitespace:
        return content.strip()
    return content


def first_pre_process(s, extrac_box=True):
    s = s.replace('\\{', '(')
    s = s.replace('\\}', ')')
    if not brackets_balanced(s):
        return s
    if extrac_box:
        boxed_content = remove_command(s, '\\boxed', keep_inside=True)
    else:
        boxed_content = s
    exist_overall_brace = True
    cnt = 0
    while exist_overall_brace and cnt < 10:
        boxed_content, exist_overall_brace = remove_overall_brace(
            boxed_content)
        cnt += 1

    if '\\quad' in boxed_content:
        boxed_content = boxed_content.split('\\quad')[0]

    last_equal_content = extract_last_equal_content(boxed_content)

    exist_overall_brace = True
    cnt = 0
    while exist_overall_brace and cnt < 10:
        last_equal_content, exist_overall_brace = remove_overall_brace(
            last_equal_content)
        cnt += 1
    return last_equal_content


def second_pre_process(s):
    kill_commands = ['\\begin', '\\end']
    remove_commands = [
        '\\text',
        '\\mathbf',
        '\\mathrm',
        '\\pmb',
        '\\hat',
        '\\overline',
        '\\boldsymbol',
    ]

    remove_content = [
        '\\,', '$', ',', '`', 'latex', '\\left', '\\right', '\\text',
        '\\mathrm', '\\Bigr', '\\Bigl', '\n', '\\]', '\\[', '\\Big', '\\bigl',
        '\\bigr', '\\biggl', '\\biggr', '\\displaystyle', '\\boldsymbol',
        '\\infty'
    ]
    replace_content = [
        ('\\operatorname{asin}', '\\asin'), ('\\operatorname{sech}', '\\sech'),
        ('\\operatorname{acos}', '\\acos'), ('\\operatorname{sinh}', '\\sinh'),
        ('\\dfrac', '\\frac'), ('\\tfrac', '\\frac'), ('\\Exp', '\\exp'),
        ('\\times', '\\bar{times}'), ('\\partial', '\\bar{partial}'),
        ('\\perp', '\\bar{perp}'), ('\\epsilon', '\\varepsilon'),
        ('\\varOmega', '\\Omega'), ('I', '\\bar{I}'), ('_e', '_{e}'),
        ('e_', '\\bar{e}_'), ('E_', '\\bar{E}_'), ('\\pm', '+'), ('\\mp', '-'),
        ('{+}', '{p}'), ('{-}', '{m}'), ('_+', '_p'), ('_-', '_m')
    ]
    for command in kill_commands:
        s = remove_command(s, command, keep_inside=False)
    for command in remove_commands:
        s = remove_command(s, command, keep_inside=True)
    for content in remove_content:
        s = s.replace(content, '')
    for content in replace_content:
        s = s.replace(content[0], content[1])
    s = convert_latex_fractions(s)
    s = bar_inside_vec(s)
    s = vec_lower_idx(s)
    s = convert_vec_syntax(s)
    s = exp_frac(s)
    if s and s[-1] == '.':
        return s[:-1]
    return s


class MyConfig:

    interpret_as_mixed_fractions: bool = False
    interpret_simple_eq_as_assignment: bool = False
    interpret_contains_as_eq: bool = True
    lowercase_symbols: bool = False


class MyNormalization:
    basic_latex: bool = True
    units: bool = False
    malformed_operators: bool = True
    nits: bool = True
    boxed = 'all'
    equations: bool = False


def master_convert(s):
    preprocessed_stage1 = first_pre_process(s)

    preprocessed_stage2 = second_pre_process(preprocessed_stage1)

    Sym = latex2sympy(preprocessed_stage2,
                      normalization_config=MyNormalization(),
                      conversion_config=MyConfig())
    return Sym


# The costs can be modified if you think their values are different
insert_cost = {'number': 1, 'symbol': 1, 'operator': 1, 'function': 1}
delete_cost = {'number': 1, 'symbol': 1, 'operator': 1, 'function': 1}
update_cost = {'number': 1, 'symbol': 1, 'operator': 1, 'function': 1}

change_type_cost = 1

bar_size = 5
discount_slope = 0.6

simplify_time_limit = 30
equals_time_limit = 10


def update_func(x, y):

    if x.label == y.label:
        return 0

    elif x.label.split('_')[0] == y.label.split('_')[0]:
        return update_cost[x.label.split('_')[0]]
    return change_type_cost


def remove_func(x):
    return delete_cost[x.label.split('_')[0]]


def remove_tree_func(x):
    if not x.children:
        return remove_func(x)
    s = calc_tree_size(x)
    return min(s, discount_slope * (s - bar_size) + bar_size)


def insert_func(x):
    return insert_cost[x.label.split('_')[0]]


def insert_tree_func(x):
    return remove_tree_func(x)


def calc_tree_size(node):

    total = insert_cost[node.label.split('_')[0]]

    if node.children and node.subtree_size != 0:

        return node.subtree_size

    for child in node.children:
        total += calc_tree_size(child)

    node.subtree_size = total

    return total


"""
Scoring function from relative distance
"""


def score_calc(tree_dist, tree_size):

    if tree_dist == 0.:
        return 100
    return max(0, 100 * discount_slope - 100 * tree_dist / tree_size)


@timeout_decorator.timeout(30, timeout_exception=TimeoutError)
def simplify_with_timeout(expr):
    return simplify(expr)


def time_simplify(expr):
    try:
        result = simplify_with_timeout(expr)
        return result
    except TimeoutError:
        return expr


@timeout_decorator.timeout(10, timeout_exception=TimeoutError)
def equal_with_timeout(expr1, expr2):
    return expr1.equals(expr2)


def time_equal(expr1, expr2):
    try:
        result = equal_with_timeout(expr1, expr2)
        return result
    except TimeoutError:
        return False


def sympy_to_tree(expr):
    """Convert the sympy expression to a tree."""
    # Symbols and constants
    if_list = [Integer, Pi, Exp1, Float, Rational, Infinity, NegativeInfinity]
    for i in if_list:
        if isinstance(expr, i):
            return TreeNode(label='number_' + str(expr), children=[])
    if isinstance(expr, (Symbol, )):

        return TreeNode(label='symbol_' + str(expr), children=[])

    # Binary operators
    elif isinstance(expr, (Add, Mul, Pow)):

        op_name = type(expr).__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label='operator_' + op_name, children=children)

    elif isinstance(expr, (Function)):
        # Functions

        func_name = expr.func.__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label='function_' + func_name, children=children)

    else:
        raise ValueError(f'Unsupported SymPy type: {type(expr)}')


class TreeNode:

    def __init__(self, label, children=None, node_type='other'):
        self.label = label
        self.children = children if children is not None else []
        self.node_type = node_type
        self.subtree_size = 0

    def get_children(self):
        return self.children

    def __str__(self):
        return self.label


def print_tree(node, indent=0):
    """Print a tree structure."""
    print('  ' * indent + f'└─ {node.label}')
    for child in node.children:
        print_tree(child, indent + 1)


class LaTeXError(Exception):

    def __init__(self, message='LaTeXError'):
        super().__init__(message)


class SymPyError(Exception):

    def __init__(self, message='SymPyError'):
        super().__init__(message)


class TreeError(Exception):

    def __init__(self, message='TreeError'):
        super().__init__(message)


class DistError(Exception):

    def __init__(self, message='DistanceError'):
        super().__init__(message)


def EED(answer_latex, test_latex, debug_mode=False):
    if not test_latex:
        return 0, -1, -1, -1
    if '\\int' in test_latex or '\\int' in answer_latex:
        return 0, -1, -1, -1
    if '\\sum' in test_latex or '\\sum' in answer_latex:
        return 0, -1, -1, 1
    if answer_latex == test_latex:
        return 100, 0.0, -1, 0
    if len(test_latex) > 3 * len(answer_latex):
        return 0, -1, -1, -1

    try:

        answer_exp = master_convert(answer_latex)
        test_exp = master_convert(test_latex)
    except Exception:
        if debug_mode:
            raise LaTeXError(f'Fail to convert latex.\n GT:{answer_latex}\n'
                             f' GEN:{test_latex}')
        return 0, -1, -1, -1

    try:

        answer_exp, rep1 = posify(answer_exp)

        answer_exp = time_simplify(answer_exp)

        test_exp, rep2 = posify(test_exp)
        test_exp = time_simplify(test_exp)

        answer_exp = answer_exp.subs(rep1)
        test_exp = test_exp.subs(rep2)

        zero_exp = time_simplify(expand(answer_exp - test_exp))

        if answer_exp == test_exp or zero_exp == 0:
            return 100, 0., 0, 0

        if time_equal(answer_exp, test_exp):
            return 100, 0., 0, 0

    except Exception:
        if debug_mode:
            raise SymPyError(
                f'Failed to simplify the sympy expression. Expressions: '
                f'answer_exp={answer_exp}, test_exp={test_exp}')
        return 0, -1, -1, -1

    try:
        tree_answer = sympy_to_tree(answer_exp)
        tree_test = sympy_to_tree(test_exp)

    except Exception:
        if debug_mode:
            raise SymPyError(f'Failed to build the sympy expression tree.\n'
                             f' GT:{answer_exp}\n GEN:{test_exp}')
        return 0, -1, -1, -1

    try:
        distance = ext_distance(tree_test,
                                tree_answer,
                                get_children=lambda x: x.get_children(),
                                single_insert_cost=insert_func,
                                insert_cost=insert_tree_func,
                                single_remove_cost=remove_func,
                                remove_cost=remove_tree_func,
                                update_cost=update_func)
    except Exception:
        if debug_mode:
            raise DistError(
                f'Failed to calculate the distance between trees.\n'
                f' GT:{answer_latex}\n GEN:{test_latex}')
        return 0, -1, calc_tree_size(tree_answer), -1
    tree_size = calc_tree_size(tree_answer)
    distance_number = distance

    rel_distance = distance / tree_size

    score = score_calc(distance_number, tree_size)
    return score, rel_distance, tree_size, distance_number
