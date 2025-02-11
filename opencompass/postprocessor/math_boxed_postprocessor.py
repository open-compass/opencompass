import re
from dataclasses import dataclass, field
from functools import lru_cache

import sympy
from latex2sympy2_extended.latex2sympy2 import (NormalizationConfig,
                                                latex2sympy, normalize_latex)


def timeout(timeout_seconds: int = 10):  # noqa: C901
    """A decorator that applies a timeout to the decorated function.

    Args:
        timeout_seconds (int): Number of seconds before timing
        out the decorated function.
            Defaults to 10 seconds.

    Notes:
        On Unix systems, uses a signal-based alarm approach which
        is more efficient as it doesn't require spawning a new process.
        On Windows systems, uses a multiprocessing-based approach
        since signal.alarm is not available. This will incur a huge
        performance penalty.
    """
    # Unix-like approach: signal.alarm
    import signal

    def decorator(func):

        def handler(signum, frame):
            raise TimeoutError('Operation timed out!')

        def wrapper(*args, **kwargs):
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_seconds)
            try:
                return func(*args, **kwargs)
            finally:
                # Cancel the alarm and restore previous handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


@dataclass(frozen=True)
class LatexExtractionConfig:
    """Config for extracting latex from the prediction.

    Attributes:
        normalization_config (NormalizationConfig):
        Configuration for LaTeX normalization.
    """

    normalization_config: NormalizationConfig = field(
        default_factory=lambda: NormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=True,
            nits=True,
            boxed='last',  # Set to "last" to extract the last boxed content
            equations=True,
        ))


@lru_cache(maxsize=1)
def lazy_latex_regex() -> list[tuple[re.Pattern[str], int]]:
    """Generate regex patterns for matching \boxed{} content."""
    # Use non-greedy matching and handle nested brackets
    boxed_pattern = r'\\boxed\{((?:[^{}]|(?:\{[^{}]*\}))*)\}'
    return [(re.compile(boxed_pattern, re.DOTALL), 0)]


@lru_cache(maxsize=20)
def parse_latex_with_timeout(latex: str, timeout_seconds: int):
    """Parse LaTeX into sympy expression."""
    return timeout(timeout_seconds)(latex2sympy)(latex,
                                                 normalization_config=None)


def extract_latex(match: re.Match, latex_config: LatexExtractionConfig,
                  timeout_seconds: int) -> tuple[sympy.Expr | str | None, str]:
    """Extract LaTeX expression from regex match."""
    try:
        # Get content of the first capture group
        latex = match.group(1)
        if not latex:
            return None, ''

        # Normalize LaTeX
        normalized_latex = normalize_latex(
            latex,
            config=latex_config.normalization_config,
        )

        try:
            # 解析为sympy表达式
            parsed_latex = parse_latex_with_timeout(
                normalized_latex, timeout_seconds=timeout_seconds)
            return parsed_latex, normalized_latex
        except Exception as e:
            print(f'Failed to parse latex: {e}')
            return None, normalized_latex
    except Exception as e:
        print(f'Failed to extract latex: {e}')
        return None, ''


def parse(
    pred: str,
    parsing_timeout: int = 3,
) -> list:
    """Extract and parse content within \boxed{} from text.

    Args:
        pred (str): Text to be parsed
        parsing_timeout (int): Parsing timeout in seconds

    Returns:
        list: List of extracted results, each result can be:
            - SymPy expression (successfully parsed mathematical expression)
            - String (original text when parsing fails)
            Returns empty list if no matches are found
    """
    try:
        # Get regex patterns
        regex_patterns = lazy_latex_regex()
        config = LatexExtractionConfig()

        extracted_predictions = []

        # Try all matches
        for pattern, _ in regex_patterns:
            for match in pattern.finditer(pred):
                extracted_match, str_fallback = extract_latex(
                    match, config, timeout_seconds=parsing_timeout)

                if extracted_match is not None:
                    extracted_predictions.append(extracted_match)
                elif str_fallback:
                    extracted_predictions.append(str_fallback)

        return extracted_predictions
    except Exception as e:
        print(f'Error during parsing: {e}')
        return []


if __name__ == '__main__':
    test_cases = [
        # 1. Basic arithmetic operations
        r'Simple fraction: \boxed{\frac{1}{2}}',
        r'Addition: \boxed{2 + 3}',
        r'Multiplication: \boxed{2 \times 3}',
        # 2. Algebraic expressions
        r'Quadratic: \boxed{x^2 + 2x + 1}',
        r'Polynomial: \boxed{3x^3 - 2x^2 + 4x - 5}',
        # 3. Trigonometric functions
        r'Trigonometry: \boxed{\sin(x) + \cos(x)}',
        r'Complex trig: \boxed{\tan^2(x) + \sec^2(x)}',
        # 4. Roots and exponents
        r'Square root: \boxed{\sqrt{16}}',
        r'Complex root: \boxed{\sqrt[3]{x^2 + 1}}',
        # 5. Logarithms
        r'Natural log: \boxed{\ln(e^2)}',
        r'Log base: \boxed{\log_2(8)}',
        # 6. Limits and summations
        r'Limit: \boxed{\lim_{x \to 0} \frac{\sin(x)}{x}}',
        r'Sum: \boxed{\sum_{i=1}^{n} i}',
        # 7. Integrals
        r'Integral: \boxed{\int_{0}^{1} x^2 dx}',
        r'Double integral: \boxed{\int_{0}^{1}\int_{0}^{1} xy \,dx\,dy}',
        # 8. Matrices
        r'Matrix: \boxed{\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}}',
        # 9. Complex combinations
        r'Complex expr: \boxed{\frac{\sqrt{x^2 + 1}}{\ln(x)} + '
        r'\int_{0}^{x} t^2 dt}',
        # 10. Error cases
        r'Empty: \boxed{}',
        r'Invalid: \boxed{\frac{1}}',  # Missing denominator
        r'Nested: \boxed{\boxed{1}}',  # Nested boxed
    ]

    def print_result(expr: str, result: list):
        print('\n' + '=' * 50)
        print(f'Input: {expr}')
        print(f'Output type: {type(result)}')
        print(f'Output: {result}')

        # If result is sympy expression, show more information
        if result:
            for item in result:
                if isinstance(item, sympy.Basic):
                    print(f'Sympy repr: {repr(item)}')
                    try:
                        print(f'Evaluated: {item.evalf()}')
                    except Exception as e:
                        print(f'Cannot evaluate: {e}')

    # Test all cases
    for test_expr in test_cases:
        try:
            result = parse(test_expr)
            print_result(test_expr, result)
        except Exception as e:
            print(f'\nError processing {test_expr}: {e}')

    # Special test: verify numerical calculations
    numerical_tests = [
        r'\boxed{2 + 2}',  # Should equal 4
        r'\boxed{\frac{1}{2} + \frac{1}{3}}',  # Should equal 5/6
        r'\boxed{\sqrt{16} + \sqrt{9}}',  # Should equal 7
    ]

    print('\n' + '=' * 50 + '\nNumerical Verification Tests:')
    for test_expr in numerical_tests:
        try:
            result = parse(test_expr)
            if result and isinstance(result[0], sympy.Basic):
                expr = result[0]
                print(f'\nExpression: {test_expr}')
                print(f'Symbolic: {expr}')
                print(f'Numerical value: {float(expr.evalf())}')
        except Exception as e:
            print(f'\nError in numerical test {test_expr}: {e}')
