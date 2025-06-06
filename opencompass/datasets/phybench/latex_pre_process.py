# flake8: noqa
#This file is used to pre-process input latex expressions
#You only need a "master_convert()"
from sympy import simplify


def brackets_balanced(s: str) -> bool:
    """
    Check if the brackets in a LaTeX string are balanced
    Args:
        s(str): the input string
    Return:
        bool: True if the brackets are balanced, False otherwise
    """
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


import re


def extract_bracket_content(s: str, bracket_position: int) -> str:
    start_idx = bracket_position

    stack = []
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

    def remove_command(s, command, keep_inside=False):
        """Removes all occurrences of a specified LaTeX-style command from a
        string.

        This function searches for a given command in the input string `s` and removes it, 
        along with its associated content enclosed in curly braces `{}`. If `keep_inside` 
        is set to `True`, the content inside the braces is preserved, and only the command 
        itself is removed. The function handles nested braces correctly.

        Args:
            s (str): The input string from which the command should be removed.
            command (str): The LaTeX-style command to be removed (e.g., "\\textbf").
            keep_inside (bool, optional): If `True`, preserves the content inside the braces 
                while removing the command. Defaults to `False`.

        Returns:
            str: The modified string with the specified command removed.

        Examples:
            >>> remove_command("This is \\textbf{bold text}.", "\\textbf")
            'This is bold text.'
            
            >>> remove_command("This is \\textbf{bold text}.", "\\textbf", keep_inside=True)
            'This is bold text.'
            
            >>> remove_command("Nested \\textbf{bold \\textit{italic text}} example.", "\\textbf")
            'Nested bold \\textit{italic text} example.'
        """

    pos = s.find(command)
    if pos < 0:
        return s
    end_index = pos + len(command)
    level = 0
    escaped = False
    #print(end_index,s[end_index])
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


import re


def convert_latex_fractions(latex_str):
    """Convert non-standard fraction like \frac\alpha2 to its standard-
    convertible \frac{\alpha}{2} We support single letter,number or standard
    form."""
    pattern = r'\\frac((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))'

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
        #print(s[final])
        if final == len(s) or not '}' in s[final + 1:]:
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
        #print(s[idx])
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
        #print(s1)

        s1 = remove_command(s1, '\\bar', keep_inside=True)
        s2 = ''.join([s[0:idx + 1], s1, s[idx2:]])
        s = s2
    return s


def vec_lower_idx(input_str):
    """in the annoying latex2sympy, error may occur when\ vec{a_{b}},we
    need\\vec{a_b} Args： input_str (str): Original string.

    Return：
        str(str): Converted
    """
    pattern = r'\\vec\{([^{}]+)_{([^{}]+)}\}'
    replacement = r'\\vec{\1}_{\2}'
    return re.sub(pattern, replacement, input_str)


def convert_vec_syntax(text):
    """Converts LaTeX vector syntax to a standardized form.

    This function processes a given text string and ensures that LaTeX vector 
    notations are consistently formatted. Specifically, it transforms instances 
    of `\vec xxx` into `\vec{xxx}`. The function handles cases where the vector 
    notation is applied to single characters, Greek letters, or LaTeX commands.

    Args:
        text (str): The input string containing LaTeX code to be processed.

    Returns:
        str: The processed string with standardized vector syntax.

    Examples:
        >>> convert_vec_syntax(r"\vec x + \vec\alpha + \vec\Gamma")
        '\\vec{x} + \\vec{\\alpha} + \\vec{\\Gamma}'
    """

    pattern = r'\\vec(\s*)(\\?[a-zA-Zα-ωΑ-Ω]+)'
    replacement = r'\\vec{\2}'
    return re.sub(pattern, replacement, text)


def remove_outer_braces(tex_str):
    """convert {base}_{subscript} to base_{subscript} Example：

    {a}_{xyz} → a_{xyz} {\theta}_{0} → \theta_{0}
    """
    pattern = r'\{(\\(?:[a-zA-Z]+|.)|[^{}])+\}_\{([^}]+)\}'
    return re.sub(pattern, r'\1_{\2}', tex_str)


def extract_last_equal_content(s: str, strip_whitespace: bool = True) -> str:
    """Extract the content after the last occurrence of specific mathematical
    comparison or assignment operators.

    :param strip_whitespace: If True, removes leading and trailing whitespace from the extracted content. Defaults to True.
    (e.g., '=', '\\approx', '\\ge', '\\le', etc.) within the input string `s`. It then extracts
    and returns the content that follows the operator. If no operator is found, the entire string
    is returned. Optionally, leading and trailing whitespace can be stripped from the extracted content.

    Args:
        s (str): The input string to process.
        strip_whitespace (bool): Whether to strip leading and trailing whitespace from the extracted content. Defaults to True.

    Returns:
        str: The content after the last matching operator, or the entire string if no operator is found.
    """
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
    """Perform the first stage of LaTeX string preprocessing.

    if not brackets_balanced(s):
        raise ValueError("The input string has unbalanced brackets. Please check the LaTeX expression.")
    equality or comparison operator.

    Args:
        s (str): The input LaTeX string to preprocess.
        extrac_box (bool): If True, extracts the content inside a '\\boxed' command. Defaults to True.

    Returns:
        str: The preprocessed LaTeX string.
    """
    #s=remove_non_ascii(s)
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
    """Perform the second stage of LaTeX string preprocessing.

    This function removes or modifies specific LaTeX commands and content to standardize
    the input string for further processing. It handles commands like '\\text', '\\mathbf',
    and '\\mathrm', removes unnecessary content, and applies transformations such as
    converting fractions and vector syntax.

    Args:
        s (str): The input LaTeX string to preprocess.

    Returns:
        str: The preprocessed LaTeX string.
    """

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
    #print(s)
    s = bar_inside_vec(s)
    s = vec_lower_idx(s)
    s = convert_vec_syntax(s)
    s = exp_frac(s)
    #s=remove_outer_braces(s)
    if s and s[-1] == '.':
        return s[:-1]
    return s


class MyConfig:

    interpret_as_mixed_fractions: bool = False
    interpret_simple_eq_as_assignment: bool = False
    interpret_contains_as_eq: bool = True
    lowercase_symbols: bool = False
    """
    Args:
        interpret_as_mixed_fractions (bool): Whether to interpret 2 \frac{1}{2} as 2/2 or 2 + 1/2
        interpret_simple_eq_as_assignment (bool): Whether to interpret simple equations as assignments k=1 -> 1
        interpret_contains_as_eq (bool): Whether to interpret contains as equality x \\in {1,2,3} -> x = {1,2,3}
        lowercase_symbols (bool): Whether to lowercase all symbols
    """


class MyNormalization:
    """Configuration for latex normalization.

    Each field controls a group of related normalizations:
    - basic_latex: Basic latex command replacements (mathrm, displaystyle, etc.)
    - units: Remove units and their variations
    - malformed_operators: Fix malformed operators (sqrt, frac, etc.)
    - nits: Small formatting fixes (spaces, dots, etc.)
    - boxed: Extract content from boxed environments
    - equations: Handle equation splitting and approximations (deprecated)
    """
    basic_latex: bool = True
    units: bool = False
    malformed_operators: bool = True
    nits: bool = True
    boxed = 'all'
    equations: bool = False


def master_convert(s):
    """The only function needed to convert a LaTeX string into a SymPy
    expression.

    Args:
        s (str): The input LaTeX string. It should be a valid LaTeX mathematical expression, 
                 such as equations, fractions, or symbols, and must have balanced brackets.

    Returns:
        Sym (Sympy Expression): A SymPy expression representing the mathematical content of the input string.
                                The returned object can be used for symbolic computation, simplification, 
                                or evaluation using SymPy's functionality.

    Example:
        >>> master_convert("\\frac{1}{2} + x")
        1/2 + x
    """
    preprocessed_stage1 = first_pre_process(s)

    preprocessed_stage2 = second_pre_process(preprocessed_stage1)
    try:
        from latex2sympy2_extended import latex2sympy
    except ImportError:
        print(
            "latex2sympy2_extended is not installed. Please install it using 'pip install latex2sympy2_extended'."
        )
        return None
    Sym = latex2sympy(preprocessed_stage2,
                      normalization_config=MyNormalization(),
                      conversion_config=MyConfig())
    return Sym
