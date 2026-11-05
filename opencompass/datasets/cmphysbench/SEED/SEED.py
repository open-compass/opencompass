# flake8: noqa
import re

# import pint  # NEW: For physical unit conversion and comparison
import timeout_decorator
from sympy import *  # noqa: F401, F403
from sympy import Derivative
from sympy.core.numbers import (ComplexInfinity, Exp1, ImaginaryUnit, Infinity,
                                NaN, NegativeInfinity, Pi)
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Not, Or
from sympy.matrices import MatrixBase
from sympy.simplify import *  # noqa: F401, F403

from .extended_zss import ext_distance
from .latex_pre_process import *  # noqa: F401, F403

# from graphviz import Digraph
"""
There are four main categories:

Constants: such as integers, decimals, or mathematical constants like π and e.
Variables: letters like x, y, z, or specified terms in problems
(e.g., ħ, c, G).
Functions: sine, cosine, exponential, logarithm, etc.
Operators: basic binary operations including addition,
multiplication, and exponentiation.
"""
# The costs can be modified if you think their values are different
insert_cost = {
    'number': 1,
    'symbol': 1,
    'operator': 1,
    'function': 1,
    'matrix': 1,
    'relation': 1
}
delete_cost = {
    'number': 1,
    'symbol': 1,
    'operator': 1,
    'function': 1,
    'matrix': 1,
    'relation': 1
}
update_cost = {
    'number': 1,
    'symbol': 1,
    'operator': 1,
    'function': 1,
    'matrix': 1,
    'relation': 1
}

# the cost of an update between different types,can be set to higher
change_type_cost = 1

bar_size = 5  # the minimum size of triggering cluster discount
discount_slope = 0.6  # discount

simplify_time_limit = 30  # set the time limit of simplify
equals_time_limit = 10  # set the time limit of equals


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
    """
    Calculate the size of a subtree based on its total insertion cost.

    The function computes the size of a subtree by summing up the insertion
    costs of the current node and all its descendant nodes. If the subtree
    size has already been calculated and stored in `node.subtree_size`, it
    returns the cached value to avoid redundant computation.

    Args:
        node (Node): The root node of the subtree for which the size is to
                     be calculated
    Returns:
        int: The total size of the subtree, calculated as the sum of the
             insertion costs of the current node and all its descendants.
    Notes:
        - The `insert_cost` dictionary is assumed to be globally defined
          and maps node labels to their respective insertion costs.
        - The function modifies the `subtree_size` attribute of the input
          node to store the calculated subtree size for future use.
    """
    """The size of a subtree equals to its total insertion cost"""

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


def numeric_score_calc(student_answer_exp, ground_truth_exp):
    """
    Specialized scoring function for numeric types
    Scores based on combined criteria of absolute and
    relative errors with configurable thresholds
    Features
    - Multi-tier scoring: 100pts (0.5% tolerance), 90pts (1%), 80pts (2%)
    - Sign consistency checking to catch conceptual errors
    - Special handling for zero values
    - Graceful fallback to tree-based scoring on conversion failures
    """
    #  Parameter Setting Section (Adjust scoring strictness)

    # 100-point standard (strictest)
    RelTol_100_strict = 0.005  # 0.5%

    # 90-point standard (moderately strict)
    RelTol_90 = 0.01  # 1%

    # 80-point standard (more lenient)
    RelTol_80 = 0.02  # 2%

    try:
        # If ground_truth_exp is an equation, extract the right-hand side value
        if hasattr(ground_truth_exp, 'rhs'):
            ground_truth_value = ground_truth_exp.rhs
            print(f'Detected equation, using rhs: {ground_truth_value}')
        else:
            ground_truth_value = ground_truth_exp

        # Try to convert SymPy expressions to numerical values
        ground_truth = float(ground_truth_value.evalf())
        student_answer = float(student_answer_exp.evalf())

        # Preprocessing: Handle special case where correct answer is 0
        if ground_truth == 0:
            if student_answer == 0:
                return 100
            else:
                return 0

        # Sign consistency check
        if ground_truth * student_answer < 0:
            return 0

        # Calculate errors
        absolute_error = abs(student_answer - ground_truth)
        relative_error = absolute_error / abs(ground_truth)

        # Judge
        is_extremely_close = (relative_error <= RelTol_100_strict)
        if is_extremely_close:
            return 100
        elif relative_error <= RelTol_90:
            return 90
        elif relative_error <= RelTol_80:
            return 80
        # None of the standards are met
        else:
            return 0

    except Exception as e:
        print(f'  -> numeric_score_calc error: {e}')
        # If numerical conversion fails,
        # fall back to the original scoring method
        return 0


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
    """
    Convert a SymPy expression into a tree structure.
    This function takes a SymPy expression and recursively
    converts it into a tree representation using `TreeNode` objects.
    Each node in the tree is labeled based on the type of
    the SymPy expression (e.g., number, symbol, operator, or function),
    and its children represent the arguments of the expression.
    Args:
        expr (sympy.Basic): The SymPy expression to be converted.
    Returns:
        TreeNode: The root node of the tree representation of the
        SymPy expression.
    Raises:
        ValueError: If the SymPy expression contains an unsupported type.
    Supported Types:
        - Numbers: Integer, Pi, Exp1, Float, Rational,
        Infinity, NegativeInfinity
        - Symbols: Symbol
        - Binary Operators: Add, Mul, Pow
        - Functions: Any subclass of `sympy.Function`
    Example:
        >>> from sympy import symbols, sin, pi
        >>> x, y = symbols('x y')
        >>> expr = x + y * sin(pi)
        >>> tree = sympy_to_tree(expr)
        >>> print(tree)
    """
    """Convert the sympy expression to a tree"""
    if isinstance(expr, MatrixBase):
        children = []
        for i in range(expr.rows):
            for j in range(expr.cols):
                children.append(sympy_to_tree(expr[i, j]))
        return TreeNode(label=f'matrix_{expr.rows}x{expr.cols}',
                        children=children)

    elif isinstance(expr, (Integer, Pi, Exp1, ImaginaryUnit, Float, Rational,
                           Infinity, NegativeInfinity, NaN, ComplexInfinity)):
        return TreeNode(label='number_' + str(expr), children=[])
    elif isinstance(expr, Symbol):
        return TreeNode(label='symbol_' + str(expr), children=[])
    elif isinstance(expr, (Add, Mul, Pow)):
        op_name = type(expr).__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label='operator_' + op_name, children=children)
    elif isinstance(expr, Function):
        func_name = expr.func.__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label='function_' + func_name, children=children)
    elif isinstance(expr, Relational):
        op_name = type(expr).__name__
        children = [sympy_to_tree(expr.lhs), sympy_to_tree(expr.rhs)]
        return TreeNode(label='relation_' + op_name, children=children)
    elif isinstance(expr, Derivative):
        children = [sympy_to_tree(expr.expr)
                    ] + [sympy_to_tree(v) for v in expr.variables]
        return TreeNode(label='function_Derivative', children=children)
    elif isinstance(expr, And):
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label='logic_And', children=children)
    elif isinstance(expr, Or):
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label='logic_Or', children=children)
    elif isinstance(expr, Not):
        children = [sympy_to_tree(expr.args[0])]
        return TreeNode(label='logic_Not', children=children)
    else:
        raise ValueError(
            f'Unsupported SymPy type: {type(expr)} Expression: {expr}')


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
    """Print a tree structure"""
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


def Equation_standardize(latex):
    """
    Standardize equation by converting it to difference form
    """
    return latex.args[0] - latex.args[1]


def extract_interval(latex):
    """
    Extract interval notation from LaTeX string
    Use regular strings (not raw strings),
    so all backslashes are escaped with \\
    """
    interval_pattern = re.compile(r'^\s*'  # Leading whitespace
                                  r'(?:\\left)?\s*'  # Optional \left
                                  r'([\(\[])\s*'  # Group 1: left bracket
                                  r'(.*?)\s*,\s*'  # Group 2: lower bound
                                  r'(.*?)\s*'  # Group 3: upper bound
                                  r'(?:\\right)?\s*'  # Optional \right
                                  r'([\)\]])\s*$'  # Group 4: right bracket
                                  )
    match = interval_pattern.match(latex)
    if match:
        left_bracket, lower_bound, upper_bound, right_bracket = match.groups()
        return True, left_bracket, lower_bound, upper_bound, right_bracket
    else:
        return False, None, None, None, None


def judge_interval(latex):
    """
    Judge if a LaTeX string represents an interval
    """
    latex = latex.replace('$', '')
    match, left_bracket, lower_bound, upper_bound, right_bracket\
        = extract_interval(latex)
    if match:
        # Judge whether it's open/closed interval
        is_left_closed = left_bracket == '['
        is_right_closed = right_bracket == ']'
        left_type = '2*' if is_left_closed else '1*'
        right_type = '*4' if is_right_closed else '*3'
        return True, left_type + lower_bound + '+' + upper_bound + right_type
    else:
        return False, latex


def check_latex_wrap(s):
    s = s.strip()
    pattern = r'''
        ^(
            \(.*\) |                            # Regular parentheses ( )
            \[.*\] |                            # Regular square brackets [ ]
            \\\(.*\\\) |                        # LaTeX inline math: \( \)
            \\\[.*\\\] |                        # LaTeX display math: \[ \]
            \\\\left\(.*\\\\right\) |           # LaTeX \left( \right)
            \\\\left\[.*\\\\right\] |           # LaTeX \left[ \right]
            \$.*\$                              # LaTeX inline math with $...$
        )$
    '''
    return re.match(pattern, s, re.VERBOSE) is not None


def parse_bracketed_string(s):
    # Remove surrounding brackets: supports (), \left( \right)
    s = s.strip()
    s = re.sub(r'^\\left\(|^\(', '', s)
    s = re.sub(r'\\right\)$|\)$', '', s)
    parts = [item.strip() for item in s.split(',')]
    return parts


def strip_dollar_signs(s):
    s = s.strip()
    if s.startswith('$$') and s.endswith('$$'):
        return s[2:-2].strip()
    elif s.startswith('$') and s.endswith('$'):
        return s[1:-1].strip()
    return s


def extract_numeric_part(latex_str: str) -> str:
    """
    Numeric extractor
    Intelligently extracts and returns a clean string containing only numbers
    and basic operators from a complex LaTeX string that may contain units,
    variables, equations.
    """
    if not isinstance(latex_str, str) or not latex_str:
        return ''

    s = latex_str.strip()

    # Strip outer LaTeX math environment delimiters
    if s.startswith('$') and s.endswith('$'):
        s = s.strip('$').strip()
    if s.startswith('\\(') and s.endswith('\\)'):
        s = s[2:-2].strip()
    if s.startswith('\\[') and s.endswith('\\]'):
        s = s[2:-2].strip()
    """
    If there's an equation or approximately equal sign,
    take only the right side
    Use non-greedy matching .*? to ensure it doesn't
    accidentally match too much
    Support various forms like a = b, a \approx b, etc.
    """
    equal_sign_pattern = r'.*?(?:=|\\approx|\\sim|\\simeq|\\propto)\s*(.*)'
    match = re.search(equal_sign_pattern, s)
    if match:
        s = match.group(1).strip()
    """
    Actively match and extract scientific notation or regular numbers
    This regex can match various forms like
    -1.28, 1.28e-5, -1.28 \times 10^{-5}, -1.28 \\times 10^{-5}, etc.
    """
    numeric_pattern = re.compile(
        r'([-+]?\s*(?:\d+\.?\d*|\.\d+)\s*(?:(?:e|E)\s*[-+]?\s*\d+|'
        r'\\\\?times\s*10\^\{?[-+]?\d+\}?)?)')

    match = numeric_pattern.search(s)

    if match:
        # If successful match, directly return the core numeric string
        numeric_part = match.group(0)
        # Clean up by replacing both \times and \\times with *
        cleaned_part = numeric_part.replace('\\\\times',
                                            '*').replace('\\times', '*')
        return cleaned_part.strip()
    else:
        return s


def extract_tuple(latex):
    """
    A tuple/key-value pair parser.
    Core strategy:
    1. If the expression is in the form `(keys) = (values)`,
    [ignore] the left `(keys) =` part,
       only take the right `(values)` as the parsing target.
    2. If the expression is just a tuple `(values)`, parse it directly.
    3. Always return a dictionary with numeric indices as keys,
    like {'0': val1, '1': val2, ...}.
    """
    latex = strip_dollar_signs(latex.strip())
    latex = latex.replace(r'\left', '')
    latex = latex.replace(r'\right', '')

    # Check if there's a top-level '(keys) = (values)' structure
    paren_level = 0
    top_level_equal_index = -1
    for i, char in enumerate(latex):
        if char in '({[':
            paren_level += 1
        elif char in ')}]':
            paren_level -= 1
        elif char == '=' and paren_level == 0:
            top_level_equal_index = i
            break
    # If found this structure,
    # we only focus on the right side of the equals sign
    if top_level_equal_index != -1:
        left_part = latex[:top_level_equal_index].strip()
        right_part = latex[top_level_equal_index + 1:].strip()
        # Do a sanity check to ensure both sides of the equals sign
        # look like tuples
        if check_latex_wrap(left_part) and check_latex_wrap(right_part):
            # override the entire expression with the right side
            latex = right_part

    # Parse the final tuple string
    if not check_latex_wrap(latex):
        return {}

    # remove brackets and split by commas
    values = parse_bracketed_string(latex)

    # If it's an empty tuple "()", values will be an empty list after parsing
    if not values:
        # Here we return empty dict, the logic in EED will handle it correctly
        return {}

    # Convert value list to dictionary with numeric indices as keys
    return {str(i): v for i, v in enumerate(values)}


def clean_latex_unit(unit_str):
    r"""
    Clean LaTeX unit string for pint parsing
    Recursively clean LaTeX wrapping like \mathrm{}, \text{},
    \operatorname{} from unit strings,
    extract plain text units while preserving braces in exponent parts.
    """
    pattern = re.compile(
        r'\\(mathrm|text|operatorname)\{([^{}]*(\{[^{}]*\}[^{}]*)*)\}')
    prev_str = None
    while prev_str != unit_str:
        prev_str = unit_str
        unit_str = pattern.sub(r'\2', unit_str)
    if unit_str.startswith('{') and unit_str.endswith('}'):
        unit_str = unit_str[1:-1]
    unit_str = unit_str.strip()
    return unit_str


def parse_latex_quantity_general(latex_str):
    r"""
    Generically parse LaTeX-formatted quantity strings
    to extract numeric values and units.
    Supports:
    - Numbers (including decimals, negative signs, scientific notation,
    and LaTeX-style scientific notation)
    - Units wrapped in \mathrm{} or \text{}, or without any wrapper
    - Removal of all LaTeX whitespace commands
    Returns: (float value, unit string)
    """
    numeric_part = extract_numeric_part(latex_str)

    try:
        numeric_clean = re.sub(r'\*\s*10\^', '*10**', numeric_part)
        number = float(eval(numeric_clean))
    except Exception as e:
        raise ValueError(
            f'Failed to compute numeric value: {numeric_clean}, error: {e}')

    original_numeric = re.search(
        r'[-+]?\s*(?:\d+\.?\d*|\.\d+)\s*(?:(?:e|E)\s*[-+]?\s*\d+|'
        r'\\\\?times\s*10\^\{?[-+]?\d+\}?)?', latex_str)
    if original_numeric:
        unit_part = latex_str[original_numeric.end():].strip()
    else:
        unit_part = ''

    unit_part = clean_latex_unit(unit_part)

    unit_part = re.sub(r'\\[ ,!quadthinse]*', '', unit_part)

    return number, unit_part


def convert_and_output_general(latex_qty1, latex_qty2, target_unit=None):
    """
    Parse two generalized LaTeX-formatted quantity strings,
    convert them to the target unit,
    and output the result.
    If target_unit is empty, convert to the unit of the first quantity.
    """
    import pint
    ureg = pint.UnitRegistry()

    n1, u1 = parse_latex_quantity_general(latex_qty1)
    n2, u2 = parse_latex_quantity_general(latex_qty2)

    q1 = n1 * ureg(u1)
    q2 = n2 * ureg(u2)

    if target_unit is None:
        target_unit = u1

    q1_converted = q1.to(target_unit)
    q2_converted = q2.to(target_unit)

    out1 = f'{q1_converted.magnitude} {target_unit}'
    out2 = f'{q2_converted.magnitude} {target_unit}'

    return out1, out2


def SEED(answer_latex, test_latex, type, debug_mode=False):
    """
    SEED (Scalable Expression Edit Distance) - Enhanced version of EED
    NEW FEATURES in SEED vs EED:
    Multi-type expression support: Expression, Equation, Tuple, Interval,
    Numeric Advanced numeric scoring with relative/absolute error thresholds
    Physical unit conversion and comparison using Pint library
    Intelligent tuple/key-value pair parsing and comparison
    Interval notation support with open/closed bracket distinction
    Improved equation standardization (A=B → A-B)
    Robust error handling

    Computes the similarity score and distance metrics between
    two LaTeX expressions.

    This function evaluates the equivalence of two mathematical
    expressions represented in LaTeX format. It uses symbolic computation
    and tree-based distance metrics to calculate a similarity score
    and other related metrics.

    Args:
        answer_latex: The latex expression of answer expression
        test_latex: The latex expression of test expression
        t: Expression type (Expression, Equation, Tuple, Interval, Numeric)
        debug_mode: Whether it raise errors or just skip it

    Returns:
        tuple: A tuple containing the following elements:
            - score (float): The similarity score between
            the two expressions (0 to 100).
            - relative_distance (float): The normalized distance
            between the two expressions.
            - answer_tree_size (int): The size of the expression
            tree for the answer.
            - distance (float): The raw distance between the two
            expression trees.

    Notes:
        - If either input contains unsupported LaTeX constructs
        (e.g., integrals or sums), the function returns default values
        indicating failure.
        - If the test expression is significantly longer than the
        answer expression, the function assumes they are not equivalent.
        - The function uses symbolic simplification and tree-based
        distance metrics to evaluate equivalence.
        - In case of errors during processing, the function returns
        default values unless `debug_mode` is enabled,
        in which case it raises specific exceptions.

    Exceptions:
        - LaTeXError: Raised when LaTeX conversion to symbolic expressions
        fails (if `debug_mode` is True).
        - SymPyError: Raised when symbolic simplification or
        tree construction fails (if `debug_mode` is True).
        - DistError: Raised when distance calculation fails
        (if `debug_mode` is True).
    """

    if not test_latex:
        return 0, -1, -1, -1
    if '\\int' in test_latex or '\\int' in answer_latex:
        return 0, -1, -1, -1
    if '\\sum' in test_latex or '\\sum' in answer_latex:
        return 0, -1, -1, 1
    if answer_latex == test_latex:
        return 100, 0.0, -1, 0
    # if len(test_latex)>3*len(answer_latex):
    #     return 0,-1,-1,-1

    try:
        if type == 'Tuple':
            answer_dict = extract_tuple(answer_latex)
            test_dict = extract_tuple(test_latex)

            if not answer_dict or not test_dict:
                return 0, -1, -1, -1

            try:
                norm_answer_dict = {
                    master_convert(k, 'Expression'): v
                    for k, v in answer_dict.items()
                }
                norm_test_dict = {
                    master_convert(k, 'Expression'): v
                    for k, v in test_dict.items()
                }
            except Exception as e:
                if debug_mode:
                    print(f'Error normalizing tuple keys: {e}')
                return 0, -1, -1, -1

            if set(norm_answer_dict.keys()) != set(norm_test_dict.keys()):
                return 0, -1, -1, -1

            scores, rel_distances, tree_sizes, distance_numbers = 0, 0, 0, 0
            size = len(norm_answer_dict)
            if size == 0:
                return 100, 0.0, 0, 0

            for sympy_key, answer_v_latex in norm_answer_dict.items():
                test_v_latex = norm_test_dict[sympy_key]

                # Recursively call to compare SEED values
                score, rel_distance, tree_size, distance_number = SEED(
                    answer_v_latex, test_v_latex, 'Expression')
                scores += score

                if rel_distance != -1:
                    rel_distances += rel_distance
                if tree_size != -1:
                    tree_sizes += tree_size
                if distance_number != -1:
                    distance_numbers += distance_number

            return (scores / size, rel_distances / size, tree_sizes / size,
                    distance_numbers / size)

        elif type == 'Interval':
            is_interval, answer_latex = judge_interval(answer_latex)
            is_interval, test_latex = judge_interval(test_latex)
            # if is_interval:t='Interval'
        elif type == 'Numeric':
            answer_no_latex = re.sub(r'\\[a-zA-Z]+', '', answer_latex)
            test_no_latex = re.sub(r'\\[a-zA-Z]+', '', test_latex)
            has_units = bool(
                re.search(r'[a-zA-Z]', answer_no_latex + test_no_latex))

            if has_units:
                try:
                    # Perform unit conversion only when a unit is present
                    answer_latex, test_latex = convert_and_output_general(
                        answer_latex, test_latex)
                except Exception as e:
                    # If unit conversion fails,
                    # skip it and process the numeric value directly
                    print(f'Unit conversion failed: {e},',
                          'proceeding with numeric extraction only')

            # Extract the numeric part
            answer_latex = extract_numeric_part(answer_latex)
            test_latex = extract_numeric_part(test_latex)

            answer_latex = re.sub(r'\\(?![a-zA-Z])', '', answer_latex)
            test_latex = re.sub(r'\\(?![a-zA-Z])', '', test_latex)

        answer_exp = master_convert(answer_latex, type)
        test_exp = master_convert(test_latex, type)
        if type == 'Equation':
            answer_exp = Equation_standardize(answer_exp)
            test_exp = Equation_standardize(test_exp)

    except Exception:
        if debug_mode:
            raise LaTeXError('Fail to convert latex.\n',
                             f'GT:{answer_latex}\n GEN:{test_latex}')
        return 0, -1, -1, -1

    try:
        if answer_exp is None or test_exp is None:
            return 0, -1, -1, -1
        answer_exp, rep1 = posify(answer_exp)
        answer_exp = time_simplify(answer_exp)

        test_exp, rep2 = posify(test_exp)
        test_exp = time_simplify(test_exp)

        answer_exp = answer_exp.subs(rep1)
        test_exp = test_exp.subs(rep2)

        # if False:
        @timeout_decorator.timeout(10, timeout_exception=TimeoutError)
        def subtract_and_simplify_with_timeout(a, b):
            if isinstance(a, Expr) and isinstance(b, Expr):
                return simplify(expand(a - b))
            elif isinstance(a, Matrix) and isinstance(b, Matrix):
                if a.shape == b.shape:
                    return simplify(expand(a - b))
                else:
                    return 1  # Matrix dimensions do not match
            else:
                return 1

        def safe_subtract_and_simplify(a, b):
            try:
                return subtract_and_simplify_with_timeout(a, b)
            except TimeoutError:
                print('  -> subtract_and_simplify timeout, returning 1')
                return 1  # Treat as unequal if a timeout occurs
            except Exception as e:
                print(f'  -> subtract_and_simplify error: {e}')
                return 1

        zero_exp = safe_subtract_and_simplify(answer_exp, test_exp)
        # zero_exp=time_simplify(expand(answer_exp-test_exp))

        if type == 'Equation':
            if (answer_exp == test_exp or zero_exp == 0
                    or answer_exp + test_exp == 0):
                return 100, 0., 0, 0

        if answer_exp == test_exp or zero_exp == 0:
            return 100, 0., 0, 0

        if time_equal(answer_exp, test_exp):
            return 100, 0., 0, 0

    except Exception:
        if debug_mode:
            raise SymPyError(
                'Failed to simplify the sympy expression.',
                f'Expressions: answer_exp={answer_exp}, test_exp={test_exp}')
        return 0, -1, -1, -1

    try:
        tree_answer = sympy_to_tree(answer_exp)
        tree_test = sympy_to_tree(test_exp)

    except Exception:
        if debug_mode:
            raise SymPyError('Failed to build the sympy expression tree.\n',
                             f'GT:{answer_exp}\n GEN:{test_exp}')
        return 0, -1, -1, -1

    distance = ext_distance(tree_test,
                            tree_answer,
                            get_children=lambda x: x.get_children(),
                            single_insert_cost=insert_func,
                            insert_cost=insert_tree_func,
                            single_remove_cost=remove_func,
                            remove_cost=remove_tree_func,
                            update_cost=update_func)

    tree_size = calc_tree_size(tree_answer)
    distance_number = distance

    rel_distance = distance / tree_size

    # If it's Numeric type, use numerical comparison logic
    if type == 'Numeric':
        score = numeric_score_calc(test_exp, answer_exp)

        return score, -1, -1, -1
    else:
        score = score_calc(distance_number, tree_size)

        return score, rel_distance, tree_size, distance_number


if __name__ == '__main__':
    # Example usage of SEED scoring
    # -----------------------------------------------------------
    # Fill in the variables below to test SEED:
    #   gt   : Ground truth LaTeX expression string
    #   pred : Model-predicted LaTeX expression string
    #   t    : Expression type (choose one of):
    #          "Expression", "Equation", "Tuple", "Interval", "Numeric"
    # -----------------------------------------------------------

    gt = r'[0,1]'  # Ground truth LaTeX expression
    pred = r'\left[0, 1 \right]'  # Predicted LaTeX expression
    type = 'Interval'  # Answer type

    score, rel_distance, tree_size, dist = SEED(gt, pred, type)

    print('\n=== Test Result ===')
    print(f'GT LaTeX:      {gt}')
    print(f'Predicted:     {pred}')
    print(f'Score:         {score}')  # Final SEED score
    print(f'Rel Distance:  {rel_distance}')  # Relative edit distance
    print(f'Tree Size:     {tree_size}'
          )  # Number of nodes in the ground truth expression tree
    print(f'Raw Distance:  {dist}')  # Raw node edit distance
