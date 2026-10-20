# flake8: noqa
# This file is used to pre-process input latex expressions
# You only need a "master_convert()"
import re

import timeout_decorator

# from latex2sympy2_extended import *  # noqa: F401, F403


def convert_caret_to_derivative(latex_str):
    # Match multiple consecutive ^ after variable names (2 or more)
    def repl(m):
        var = m.group(1)
        carets = m.group(2)
        n = len(carets)
        if n == 2:
            return f"{var}''"  # Second order uses double prime notation
        else:
            return f'{var}^{{({n})}}'  # Higher orders use ^{(n)} notation

    pattern = r'([a-zA-Z]+)(\^{2,})'
    return re.sub(pattern, repl, latex_str)


def preprocess_special_superscripts(latex_str):
    # Define general variable pattern:
    # variable name + optional subscript + optional existing superscript
    var_pattern = r'([a-zA-Z0-9_\\]+(?:_\{[^}]+\})?(?:\^\{[^}]+\})?)'

    # 1. Replace ^+ -> ^{+}
    latex_str = re.sub(fr'{var_pattern}\^\+', r'\1^{+}', latex_str)

    # 2. Replace ^- -> ^{-}
    latex_str = re.sub(fr'{var_pattern}\^\-', r'\1^{-}', latex_str)

    # 3. Replace ^* -> ^{star}
    latex_str = re.sub(fr'{var_pattern}\^\*', r'\1^{star}', latex_str)
    latex_str = re.sub(fr'{var_pattern}\^\{{(\\ast|\*)\}}', r'\1^{star}',
                       latex_str)
    latex_str = re.sub(r'\^\{(\\ast|\*)\}', r'^{star}', latex_str)
    # 4. Replace invalid empty exponents with ^{prime}
    latex_str = re.sub(fr'{var_pattern}\^(?![\{{\\a-zA-Z0-9])', r'\1^{prime}',
                       latex_str)

    return latex_str


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
    """Remove non-ASCII characters from text"""
    return text.encode('ascii', errors='ignore').decode()


def extract_bracket_content(s: str, bracket_position: int) -> str:
    """Extract content within braces starting from given position"""
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
    """Find the position of the first unescaped opening brace"""
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
    """extract the command name from a bracket"""
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
    """
    Removes all occurrences of a specified LaTeX-style command
    from a string using an iterative approach.

    This function is more robust and efficient than a recursive
    solution, avoiding recursion depth limits and excessive string copying.

    Args:
        s (str): The input string.
        command (str): The LaTeX-style command to remove (e.g., "\\textbf").
        keep_inside (bool, optional):
        If True, keeps the content inside the braces.
        Defaults to False.

    Returns:
        str: The modified string.

    Examples:
        >>> remove_command("This is \\textbf{bold text}.", "\\textbf")
        'This is '
        >>> remove_command("This is \\textbf{bold text}.",
        "\\textbf", keep_inside=True)
        'This is bold text.'
        >>> remove_command("Nested
        \\textbf{bold \\textit{italic text}} example.",
        "\\textbf", keep_inside=True)
        'Nested bold \\textit{italic text} example.'
        >>> remove_command("No braces \\here.", "\\here")
        'No braces .'
        >>> remove_command("Mismatched \\textbf{braces", "\\textbf")
        'Mismatched \\textbf{braces' # No replacement if brace is not closed
    """
    result_parts = []
    current_pos = 0
    while True:
        pos = s.find(command, current_pos)

        # If no more commands are found, end the loop
        if pos == -1:
            result_parts.append(s[current_pos:])
            break

        # 1. Add the part before the command
        result_parts.append(s[current_pos:pos])

        # Find the first character after the command, check if it's '{'
        brace_start_pos = pos + len(command)

        if brace_start_pos < len(s) and s[brace_start_pos] == '{':
            # Find the matching '}'
            level = 0
            brace_end_pos = -1
            for i in range(brace_start_pos, len(s)):
                if s[i] == '{':
                    level += 1
                elif s[i] == '}':
                    level -= 1
                    if level == 0:
                        brace_end_pos = i
                        break

            if brace_end_pos != -1:  # Successfully found matching bracket
                if keep_inside:
                    # Keep the content inside the brackets
                    result_parts.append(s[brace_start_pos + 1:brace_end_pos])
                # Update next search start position,
                # skip the entire command and its content
                current_pos = brace_end_pos + 1
            else:  # No matching bracket found, don't process
                # Add the command itself back, then start
                # searching from after the command
                result_parts.append(s[pos:brace_start_pos + 1])
                current_pos = brace_start_pos + 1

        else:  # No bracket after command, only remove the command itself
            current_pos = brace_start_pos

    return ''.join(result_parts)


def convert_latex_fractions(latex_str):
    """Convert non-standard fractions to standard format"""
    pattern = (r'\\frac((?:\\[a-zA-Z]+|\d|[a-zA-Z]|'
               r'{[^{}]*}))((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))')

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
    """ Find the position of the first unescaped opening brace
    and extract the command before it """
    brace_pos = find_first_unescaped_brace(s)
    if brace_pos == -1:
        return None
    return extract_command(s, brace_pos)


def remove_overall_brace(s: str) -> str:
    """Remove the outermost brace pair if it wraps the entire string"""
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
    """Add braces around exponentiated fractions"""

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
    """Find all occurrences of substring in string"""
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
    """Handle bar notation inside vector commands"""
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
    """
    Args:
        input_str (str): Original string

    Returns:
        str: Converted string
    """
    pattern = r'\\vec\{([^{}]+)_{([^{}]+)}\}'
    replacement = r'\\vec{\1}_{\2}'
    return re.sub(pattern, replacement, input_str)


def convert_vec_syntax(text):
    """
    Converts LaTeX vector syntax to a standardized form.

    This function processes a given text string and ensures that LaTeX vector
    notations are consistently formatted. Specifically, it transforms instances
    of `\vec xxx` into `\vec{xxx}`. The function handles cases where the vector
    notation is applied to single characters, Greek letters, or LaTeX commands.

    Args:
        text (str): The input string containing LaTeX code to be processed.

    Returns:
        str: The processed string with standardized vector syntax.

    Examples:
        >>> convert_vec_syntax(r"\\vec x + \\vec\\alpha + \\vec\\Gamma")
        '\\vec{x} + \\vec{\\alpha} + \\vec{\\Gamma}'
    """

    pattern = r'\\vec(\s*)(\\?[a-zA-Zα-ωΑ-Ω]+)'
    replacement = r'\\vec{\2}'
    return re.sub(pattern, replacement, text)


def remove_outer_braces(tex_str):
    """
    Convert {base}_{subscript} to base_{subscript}
    Example
    {a}_{xyz} → a_{xyz}
    {\theta}_{0} → \theta_{0}
    """

    pattern = r'\{(\\(?:[a-zA-Z]+|.)|[^{}])+\}_\{([^}]+)\}'
    return re.sub(pattern, r'\1_{\2}', tex_str)


def extract_last_equal_content(s: str, strip_whitespace: bool = True) -> str:
    """
    Extract the content after the last occurrence of specific
    mathematical comparison or assignment operators.

    :param strip_whitespace: If True, removes leading and trailing whitespace
    from the extracted content. Defaults to True.
    (e.g., '=', '\\approx', '\\ge', '\\le', etc.) within the input string `s`.
    It then extracts and returns the content that follows the operator.
    If no operator is found, the entire string is returned. Optionally,
    leading and trailing whitespace can be stripped from the extracted content.

    Args:
        s (str): The input string to process.
        strip_whitespace (bool): Whether to strip leading and
        trailing whitespace from the extracted content. Defaults to True.

    Returns:
        str: The content after the last matching operator,
        or the entire string if no operator is found.
    """
    comparison_operators = ('\\approx', '\\ge', '\\le', '\\geq', '\\leq', '=')
    # '\\approx','\\ge','\\le','\\geq','\\leq','<','>',
    content = s
    for sign in comparison_operators:
        if sign in s:
            rfind_index = s.rfind(sign)
            if s[rfind_index:rfind_index + 5] == '\\left' and sign == '\\le':
                continue
            if rfind_index != -1:
                content = s[rfind_index + 1:]
                if content == '0':
                    print('')
    if strip_whitespace:
        return content.strip()
    return content


def first_pre_process(s, t, extract_box=True):
    """
    Perform the first stage of LaTeX string preprocessing.

    if not brackets_balanced(s):
        raise ValueError("The input string has unbalanced brackets.
        Please check the LaTeX expression.")
    equality or comparison operator.

    Args:
        s (str): The input LaTeX string to preprocess.
        extract_box (bool): If True, extracts the content
        inside a '\\boxed' command. Defaults to True.

    Returns:
        str: The preprocessed LaTeX string.
    """
    # s=remove_non_ascii(s)
    s = s.replace('\\{', '(')
    s = s.replace('\\}', ')')

    if t == 'Expression' or t == 'Equation':
        s = s.replace('\\approx', '=')

    if not brackets_balanced(s):
        return s
    if extract_box:
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

    if t == 'Equation':
        last_equal_content = boxed_content
    else:
        last_equal_content = extract_last_equal_content(boxed_content)

    # last_equal_content=extract_last_equal_content(boxed_content)

    exist_overall_brace = True
    cnt = 0
    while exist_overall_brace and cnt < 10:
        last_equal_content, exist_overall_brace = remove_overall_brace(
            last_equal_content)
        cnt += 1
    return last_equal_content


def remove_text_from_latex(expr: str) -> str:
    """Replace Chinese characters with '1' characters"""

    def repl(match):
        length = len(match.group())
        return '1' * length

    return re.sub(r'[\u4e00-\u9fa5]+', repl, expr)


def extract_bracket_subscript_pairs(expr):
    """Extract bracket-subscript pairs from expression"""
    matches = []
    stack = []
    i = 0
    n = len(expr)

    while i < n:
        if expr[i] in '({[':
            stack.append((i, expr[i]))
        elif expr[i] in ')}]':
            if not stack:
                i += 1
                continue
            start, open_br = stack.pop()
            close_br = expr[i]
            if (open_br, close_br) not in [('(', ')'), ('[', ']'), ('{', '}')]:
                i += 1
                continue

            j = i + 1
            if j < n and expr[j] == '_':
                k = j + 1
                if k < n and expr[k] == '{':
                    k += 1
                    while k < n and expr[k] != '}':
                        k += 1
                    k += 1
                else:
                    k += 1
                matches.append((start, k, expr[start:k]))
        i += 1
    return matches


def add_number_to_bracket_subscripts(expr):
    """Add numbering to bracket subscripts"""
    matches = extract_bracket_subscript_pairs(expr)
    if not matches:
        return expr

    matches.sort(reverse=True)
    counter = 1
    for start, end, content in matches:
        new_content = re.sub(r'(_)', f'{counter}\\1', content, count=1)
        expr = expr[:start] + new_content + expr[end:]
        counter += 1
    return expr


def insert_multiplication_symbols(expr):
    """
    Automatically insert \\cdot in LaTeX expressions where needed,
    handling implicit multiplication cases.
    Example: \\frac{1}{2}\\bar{E}1_a^i →
    \\frac{1}{2} \\cdot \\bar{E} \\cdot 1_a^i
    """

    # Add \cdot after \frac{...}{...} if directly
    # followed by variables or functions
    expr = re.sub(r'(\\frac\{[^}]+\}\{[^}]+\})(?=\\[a-zA-Z]|[a-zA-Z0-9])',
                  r'\1 \\cdot ', expr)

    # Insert \cdot between a symbol (like \bar{E}) and another variable
    expr = re.sub(r'(\})((\d|[a-zA-Z])_?[a-zA-Z]?\^?[a-zA-Z]?)',
                  r'\1 \\cdot \2', expr)

    return expr


def remove_all_text_commands(latex_str):
    """
    Remove all \text{...} commands and their content from LaTeX.
    Args:
        latex_str (str): Input LaTeX string
    Returns:
        str: String after removing \text{...}
    """
    pattern = r'\\text\{[^{}]*\}'
    return re.sub(pattern, '1', latex_str)


def convert_general_exp_format(latex_str):
    # Match patterns like x^{*2}, f(x)^{*3}, \alpha^{*4}, etc.
    pattern = r'([a-zA-Z\\]+|\([^)]+\)|\{[^}]+\})\^\{\*(\d+)\}'

    # Convert to (base^*)^n format
    return re.sub(pattern, r'(\1^*)^\2', latex_str)


def modify_latex_expression(expr: str) -> str:
    # Replace V_{CKM}^{ji*} with V_{CKM}^ji^*
    expr = re.sub(r'V_\{CKM\}\^\{([^\}]*?)\*\}', r'V_{CKM}^\1', expr)

    # Remove + appearing before \text
    expr = re.sub(r'\+\s*(\\text)', r'\1', expr)

    return expr


def wrap_single_subscripts(s: str) -> str:
    """
    Convert subscripts like xxx_Y or xxx_y to xxx_{Y}/xxx_{y}.

    - Only handle single English letters
    - If subscript is already _{...} or followed by \\command, don't modify
    """
    # Negative lookahead (?![{\\]):
    # exclude _{ already bracketed and _\command cases
    pattern = re.compile(r'_(?![{\\])([A-Za-z])')
    return pattern.sub(r'_{\1}', s)


def replace_hc_text(s: str) -> str:
    """
    Replace \text{h.c.} (case and space insensitive) with h_c,
    keep other \text{...} unchanged.
    """
    pattern = re.compile(r'\\text\s*{([^{}]*)}')

    def repl(m):
        content = m.group(1).strip()
        norm = content.lower().replace(' ', '')
        if norm in ('h.c.', 'h.c'):
            return 'h_c'
        return m.group(0)

    return pattern.sub(repl, s)


def standardize_dE_notation(s: str) -> str:
    s = re.sub(r'd\*([A-Z])_({?[a-zA-Z0-9]+}?)', r'd{\1}_\2', s)
    return s


def replace_arrow_expression(s: str) -> str:
    """
    Replace W(i arrow f) with W(iRf), i.e.,
    change 'i arrow f' to 'iRf' in parentheses.
    """
    return re.sub(r'W\(\s*(\w+)\s+arrow\s+(\w+)\s*\)', r'W(\1R\2)', s)


def preprocess_feynman_slash(latex_str: str) -> str:
    """
    Converts Feynman slash notation like \not{k} into a plain
    variable `kslash`.
    This helps latex2sympy to parse specialized physics notations.
    Example: \not{k}_0 -> kslash_0
    """
    pattern = r'\\not\{([^{}]+)\}'

    replacement = r'\\bar{\1slash}'

    return re.sub(pattern, replacement, latex_str)


def fix_subscript_on_parentheses(s: str) -> str:
    # Match pattern: (content)_{subscript}
    pattern = r'\(([^)]+)\)_\{([^}]+)\}'

    # Replacement rule: keep only "content" and "subscript", remove outer ()
    replacement = r'\1_{\2}'

    return re.sub(pattern, replacement, s)


def reorder_super_sub(latex_str: str) -> str:
    """
    Reorder base^{super}_{sub} form to base_{sub}^{super}.
    Example: M^{-1}_{j_1 i_1} -> M_{j_1 i_1}^{-1}
    This function can handle single letters, multiple letters,
    and LaTeX commands as base symbols.
    """
    # Pattern: (base symbol)(superscript)(subscript)
    # Base symbol: one or more letters, possibly starting with backslash
    # Superscript: ^{...}
    # Subscript: _{...}
    pattern = r'([a-zA-Z\\]+)(\^\{[^}]+\})(_\{[^}]+\})'
    replacement = r'\1\3\2'

    # Continuously apply replacement until the string no longer changes
    # This is a safer approach for handling more complex cases
    # (though not needed in this example)
    while True:
        new_str = re.sub(pattern, replacement, latex_str)
        if new_str == latex_str:
            break
        latex_str = new_str

    return latex_str


def second_pre_process(s):
    """
    Perform the second stage of LaTeX string preprocessing.

    This function removes or modifies specific LaTeX commands
    and content to standardize the input string for further processing.
    It handles commands like '\\text', '\\mathbf',
    and '\\mathrm', removes unnecessary content,
    and applies transformations such as
    converting fractions and vector syntax.

    Args:
        s (str): The input LaTeX string to preprocess.

    Returns:
        str: The preprocessed LaTeX string.
    """

    s = reorder_super_sub(s)

    kill_commands = ['\\begin', '\\end']
    remove_commands = [
        '\\text',
        '\\mathbf',
        '\\mathrm',
        '\\mathscr',
        '\\mathcal',
        '\\mathfrak',
        '\\pmb',
        '\\hat',
        '\\overline',
        '\\boldsymbol',
        '\\mathbb',
    ]

    remove_content = [
        '\\,', '$', ',', '`', 'latex', '\\left', '\\right', '\\text',
        '\\mathrm', '\\Bigr', '\\Bigl', '\n', '\\]', '\\[', '\\Big', '\\bigl',
        '\\bigr', '\\biggl', '\\biggr', '\\displaystyle', '\\boldsymbol',
        '\\infty'
    ]
    replace_content = [
        ('\\operatorname{asin}', '\\asin'),
        ('\\operatorname{sech}', '\\sech'),
        ('\\operatorname{acos}', '\\acos'),
        ('\\operatorname{sinh}', '\\sinh'),
        ('\\operatorname{rot}', '\\bar{rot}'),
        ('\\dfrac', '\\frac'),
        ('\\tfrac', '\\frac'),
        ('\\Exp', '\\exp'),
        ('\\gg', '>'),
        ('\\ll', '<'),
        ('\\times', '\\bar{times}'),
        ('\\dagger', '\\bar{dagger}'),
        ('\\operatorname{dim}', '\\bar{dim}'),
        ('\\overleftarrow', '\\bar{overleftarrow}'),
        (';', '\\bar{CD}'),
        ('\\partial', '\\bar{partial}'),
        ('\\perp', '\\bar{perp}'),
        ('\\parallel', '\\bar{parallel}'),
        ('\\|', '\\bar{parallel}'),
        ('\\epsilon', '\\varepsilon'),
        ('\\varOmega', '\\Omega'),
        ('I', '\\bar{I}'),
        ('_e', '_{e}'),
        ('e_', '\\bar{e}_'),
        ('E_', '\\bar{E}_'),
        ('\\pm', '+'),
        ('\\mp', '-'),
        ('{+}', '{p}'),
        ('{-}', '{m}'),
        ('_+', '_p'),
        ('_-', '_m'),
        # ('\\infty', 'oo')
    ]

    # More precise handling of single quotes:
    # distinguish derivatives and physics symbols
    # Handle function derivatives: f'(x) -> f^{prime}(x)
    s = re.sub(r'([a-zA-Z]+)\'(?=\()', r'\1^{prime}', s)
    s = re.sub(r'([a-zA-Z]+)\'(?=\s|$|[^a-zA-Z(])', r'\1^{prime}', s)
    # Handle single quotes in braces: {k}' -> {k}^{prime}
    s = re.sub(r'(\{[a-zA-Z]+\})\'', r'\1^{prime}', s)
    s = re.sub(r'·', '', s)
    # s = s.replace(r'\dagger', 'dagger')
    # s = re.sub(r'\|(.+?)\\rangle', r'\1', s)
    s = s.replace(r'\operatorname{Im}', 'Im')
    # Remove angle brackets from Dirac symbols or inner product symbols
    s = re.sub(r'\\langle\s*(.+?)\s*\\rangle', r'{\1}', s)
    s = re.sub(r'\|\s*(.+?)\s*\\rangle', r'\1', s)
    s = s.replace(r'\sim', 'Symbol("sim")')
    s = re.sub(r'\\bar\{([^{}]+)\}', r'\1', s)
    s = replace_hc_text(s)
    s = convert_general_exp_format(s)
    s = convert_caret_to_derivative(s)
    s = preprocess_special_superscripts(s)
    s = wrap_single_subscripts(s)
    s = modify_latex_expression(s)
    s = remove_all_text_commands(s)
    s = fix_subscript_on_parentheses(s)

    # s=remove_outer_braces(s)
    # Special case: protect differential forms,
    # avoid E_ replacement affecting dE_{k}
    # Handle normal form: dE_{k}
    s = re.sub(r'\bd([A-Z])_', r'd\1UNDERSCORE', s)
    # Handle mathbf form: d\mathbf{E}_{k}
    s = re.sub(r'\bd\\mathbf\{([A-Z])\}_', r'd\\mathbf{\1}UNDERSCORE', s)

    s = re.sub(r'\\ddot\{([^}]+)\}', r'\1_{ddot}', s)
    s = re.sub(r'\\ddot([A-Za-z]+)', r'\1_{ddot}', s)
    # Similarly handle \dot
    s = re.sub(r'\\dot\{([^}]+)\}', r'\1_{dot}', s)
    s = re.sub(r'\\dot([A-Za-z]+)', r'\1_{dot}', s)
    # If the string contains matrix environment keywords,
    # skip kill_commands processing
    if not ('\\begin{pmatrix}' in s or '\\end{pmatrix}' in s
            or '\\begin{bmatrix}' in s or '\\end{bmatrix}' in s
            or '\\begin{matrix}' in s or '\\end{matrix}' in s
            or '\\begin{vmatrix}' in s or '\\end{vmatrix}' in s
            or '\\begin{Vmatrix}' in s or '\\end{Vmatrix}' in s):

        for command in kill_commands:
            s = remove_command(s, command, keep_inside=False)
    for command in remove_commands:
        s = remove_command(s, command, keep_inside=True)
    for content in remove_content:
        s = s.replace(content, '')
    for content in replace_content:
        s = s.replace(content[0], content[1])
    # Restore protected differential forms and
    # add multiplication signs for latex2sympy recognition
    if '\\lim' in s:
        s = s.replace(r'arrow', r'\rightarrow')
    else:
        s = re.sub(r'\barrow\b', r'\\bar{arrow}', s)
    s = re.sub(r'd([A-Z])UNDERSCORE', r'd*\1_', s)
    s = re.sub(r'd\\mathbf\{([A-Z])\}UNDERSCORE', r'd*\\mathbf{\1}_', s)
    s = preprocess_feynman_slash(s)
    s = convert_latex_fractions(s)
    s = standardize_dE_notation(s)
    # s = replace_arrow_expression(s)
    s = bar_inside_vec(s)
    s = vec_lower_idx(s)
    s = convert_vec_syntax(s)
    s = exp_frac(s)
    if s and s[-1] == '.':
        return s[:-1]
    s = s.replace(r'\varkappa', r'\kappa')
    # First replace derivative forms to avoid parsing errors
    s = replace_derivative_frac_preserve_frac(s)
    s = remove_text_from_latex(s)
    s = add_parentheses_to_d(s)
    s = add_number_to_bracket_subscripts(s)
    s = insert_multiplication_symbols(s)
    s = s.replace('Å', 'A')
    return s


def add_parentheses_to_d(expr):
    """
    Pattern: match a 'd', but ensure it's not preceded by \frac{
    (?<!\\frac{) is "negative lookbehind assertion", requiring that
    the match position cannot be preceded by "\frac{"
    The \\ in (?<!...) needs to be escaped, so it's (?<!\\frac{)
    """
    pattern = r'(?<!\\frac{)d(\\[A-Za-z0-9_]+)'

    # Replacement rule unchanged
    return re.sub(pattern, r'd(\1)', expr)


class MyConfig:
    interpret_as_mixed_fractions: bool = False
    interpret_simple_eq_as_assignment: bool = False
    interpret_contains_as_eq: bool = True
    lowercase_symbols: bool = False
    """
    Args:
        interpret_as_mixed_fractions (bool): Whether to interpret
        2 \frac{1}{2} as 2/2 or 2 + 1/2
        interpret_simple_eq_as_assignment (bool): Whether to interpret
        simple equations as assignments k=1 -> 1
        interpret_contains_as_eq (bool): Whether to interpret contains
        as equality x \\in {1,2,3} -> x = {1,2,3}
        lowercase_symbols (bool): Whether to lowercase all symbols
    """


class MyNormalization:
    """Configuration for latex normalization.

    Each field controls a group of related normalizations:
    - basic_latex: Basic latex command replacements
    (mathrm, displaystyle, etc.)
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


def replace_derivative_frac_preserve_frac(expr: str) -> str:
    """
    Convert d<var> in \frac{d<var1>}{d<var2>} to symbol names,
    preserve \frac structure, preserve underscores _.
    """
    pattern = r'''
        \\frac\{
            d
            (\\?[a-zA-Z]+)
            (_\{?[a-zA-Z0-9]+\}?)?
        \}\{
            d
            (\\?[a-zA-Z]+)
            (_\{?[a-zA-Z0-9]+\}?)?
        \}
    '''

    def clean(s):
        return s.replace('\\', '').replace('{', '').replace('}', '')

    def repl(m):
        var1 = clean(m.group(1))
        sub1 = clean(m.group(2) or '')
        var2 = clean(m.group(3))
        sub2 = clean(m.group(4) or '')

        return f'\\frac{{D{var1}{sub1}}}{{D{var2}{sub2}}}'

    return re.sub(pattern, repl, expr, flags=re.VERBOSE)


@timeout_decorator.timeout(10, timeout_exception=TimeoutError)
def master_convert_with_timeout(s, t):
    from latex2sympy2_extended import latex2sympy
    """Master convert with timeout protection"""
    s = re.sub(r'~', '', s)
    preprocessed_stage1 = first_pre_process(s, t)
    preprocessed_stage2 = second_pre_process(preprocessed_stage1)
    Sym = latex2sympy(preprocessed_stage2,
                      normalization_config=MyNormalization(),
                      conversion_config=MyConfig())
    return Sym


def master_convert(s, t):
    """
    The only function needed to convert a LaTeX string into a SymPy expression.

    Args:
        s (str): The input LaTeX string. It should be a valid LaTeX
        mathematical expression, such as equations, fractions,
        or symbols, and must have balanced brackets.

    Returns:
        Sym (Sympy Expression): A SymPy expression representing
        the mathematical content of the input string.
        The returned object can be used for symbolic computation,
        simplification, or evaluation using SymPy's functionality.

    Example:
        >>> master_convert("\\frac{1}{2} + x")
        1/2 + x
    """
    try:
        return master_convert_with_timeout(s, t)
    except TimeoutError:
        print(f'  -> master_convert timeout for LaTeX: {s[:100]}...')
        return None
    except Exception as e:
        print(f'  -> master_convert error: {e}')
        return None
