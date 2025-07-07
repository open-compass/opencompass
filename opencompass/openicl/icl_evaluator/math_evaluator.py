from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


@ICL_EVALUATORS.register_module()
class MATHEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        try:
            from latex2sympy2_extended import NormalizationConfig
            from math_verify import (ExprExtractionConfig,
                                     LatexExtractionConfig, parse, verify)
        except ImportError:
            raise ImportError('Failed to import required modules. Please '
                              'install the necessary packages: '
                              'pip install math_verify latex2sympy2_extended')

        self.is_num_equal(predictions, references)

        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            count += 1
            j_with_env = f'${j}$'
            gold_parsed = parse(
                j_with_env,
                extraction_mode='first_match',
                extraction_config=[
                    LatexExtractionConfig(),
                    ExprExtractionConfig(),
                ],
            )

            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct
                # latex (no malformed operators)
                answer_parsed = parse(
                    i,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed='all',
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )

                answer_correct = float(verify(answer_parsed, gold_parsed))
                correct += answer_correct
                detail = {
                    'pred': str(answer_parsed),
                    'answer': str(gold_parsed),
                    'correct': True if answer_correct else False,
                }
                details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result


if __name__ == '__main__':
    import sympy
    from math_verify import parse
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
