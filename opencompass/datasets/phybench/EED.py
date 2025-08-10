# flake8: noqa
# isort: noqa
# yapf: noqa

import timeout_decorator
from sympy import *
from sympy.core.numbers import Exp1, Infinity, NegativeInfinity, Pi
from sympy.simplify import *

from opencompass.datasets.phybench.extended_zss import ext_distance
from opencompass.datasets.phybench.latex_pre_process import *

guide = """
Guide:
You only need to use EED and install the following packages:
- sympy
- numpy
- latex2sympy2_extended
- timeout_decorator
"""
"""
There are four main categories:

Constants: such as integers, decimals, or mathematical constants like π and e.
Variables: letters like x, y, z, or specified terms in problems (e.g., ħ, c, G).
Functions: sine, cosine, exponential, logarithm, etc.
Operators: basic binary operations including addition, multiplication, and exponentiation.
"""

# The costs can be modified if you think their values are different
insert_cost = {'number': 1, 'symbol': 1, 'operator': 1, 'function': 1}
delete_cost = {'number': 1, 'symbol': 1, 'operator': 1, 'function': 1}
update_cost = {'number': 1, 'symbol': 1, 'operator': 1, 'function': 1}

change_type_cost = 1  #the cost of an update between different types,can be set to higher

bar_size = 5  # the minimum size of triggering cluster discount
discount_slope = 0.6  #discount

simplify_time_limit = 30  #set the time limit of simplify
equals_time_limit = 10  #set the time limit of equals


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
    This function takes a SymPy expression and recursively converts it into a tree
    representation using `TreeNode` objects. Each node in the tree is labeled based
    on the type of the SymPy expression (e.g., number, symbol, operator, or function),
    and its children represent the arguments of the expression.
    Args:
        expr (sympy.Basic): The SymPy expression to be converted.
    Returns:
        TreeNode: The root node of the tree representation of the SymPy expression.
    Raises:
        ValueError: If the SymPy expression contains an unsupported type.
    Supported Types:
        - Numbers: Integer, Pi, Exp1, Float, Rational, Infinity, NegativeInfinity
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
    #print(expr)
    """Convert the sympy expression to a tree"""
    # Symbols and constants
    if isinstance(
            expr,
        (Integer, Pi, Exp1, Float, Rational, Infinity, NegativeInfinity)):
        return TreeNode(label='number_' + str(expr), children=[])
    elif isinstance(expr, (Symbol, )):

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
        #print(expr)
        print(
            f'Unsupported Sympy type: {type(expr).__name__}, Expression: {expr}'
        )
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


import timeout_decorator


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
    """Computes the similarity score and distance metrics between two LaTeX
    expressions. This function evaluates the equivalence of two mathematical
    expressions represented in LaTeX format. It uses symbolic computation and
    tree-based distance metrics to calculate a similarity score and other
    related metrics.

        tuple: A tuple containing the following elements:
            - score (float): The similarity score between the two expressions (0 to 100).
            - relative_distance (float): The normalized distance between the two expressions.
            - answer_tree_size (int): The size of the expression tree for the answer.
            - distance (float): The raw distance between the two expression trees.
    Notes:
        - If either input contains unsupported LaTeX constructs (e.g., integrals or sums),
          the function returns default values indicating failure.
        - If the test expression is significantly longer than the answer expression,
          the function assumes they are not equivalent.
        - The function uses symbolic simplification and tree-based distance metrics to
          evaluate equivalence.
        - In case of errors during processing, the function returns default values unless
          `debug_mode` is enabled, in which case it raises specific exceptions.
    Exceptions:
        - LaTeXError: Raised when LaTeX conversion to symbolic expressions fails (if `debug_mode` is True).
        - SymPyError: Raised when symbolic simplification or tree construction fails (if `debug_mode` is True).
        - DistError: Raised when distance calculation fails (if `debug_mode` is True).
    Args:
        answer_latex: the latex expression of answer expression
        test_latex: the latex expression of test expression
        debug_mode: whether it raise errors or just skip it
    Returns:
         tuple: A tuple containing the following elements:
            - score (float): The similarity score between the two expressions (0 to 100).
            - relative_distance (float): The normalized distance between the two expressions.
            - answer_tree_size (int): The size of the expression tree for the answer.
            - distance (float): The raw distance between the two expression trees.
    """

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
    except:
        print(
            f'Failed to convert input latex to sympy expression,please check it'
        )
        if debug_mode:
            raise LaTeXError(
                f'Fail to convert latex.\n GT:{answer_latex}\n GEN:{test_latex}'
            )
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

    except:
        print('Something happened during simplification,returning zero')
        if debug_mode:
            raise SymPyError(
                f'Failed to simplify the sympy expression. Expressions: answer_exp={answer_exp}, test_exp={test_exp}'
            )
        return 0, -1, -1, -1

    try:
        tree_answer = sympy_to_tree(answer_exp)
        tree_test = sympy_to_tree(test_exp)

    except:

        print('Failed to build expression tree,returning zero')
        if debug_mode:
            raise SymPyError(
                f'Failed to build the sympy expression tree.\n GT:{answer_exp}\n GEN:{test_exp}'
            )
        return 0, -1, -1, -1

    distance = ext_distance(tree_test,
                            tree_answer,
                            get_children=lambda x: x.get_children(),
                            single_insert_cost=insert_func,
                            insert_cost=insert_tree_func,
                            single_remove_cost=remove_func,
                            remove_cost=remove_tree_func,
                            update_cost=update_func)
    try:

        distance = ext_distance(tree_test,
                                tree_answer,
                                get_children=lambda x: x.get_children(),
                                single_insert_cost=insert_func,
                                insert_cost=insert_tree_func,
                                single_remove_cost=remove_func,
                                remove_cost=remove_tree_func,
                                update_cost=update_func)
    except:
        print('Failed to calculate distance')
        if debug_mode:
            raise DistError(
                f'Failed to calculate the distance between trees.\n GT:{answer_latex}\n GEN:{test_latex}'
            )
        return 0, -1, calc_tree_size(tree_answer), -1
    tree_size = calc_tree_size(tree_answer)
    distance_number = distance

    rel_distance = distance / tree_size

    score = score_calc(distance_number, tree_size)

    return score, rel_distance, tree_size, distance_number
