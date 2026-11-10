# flake8: noqa

import multiprocessing

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


def _verify_worker(pred, ref, queue):
    """Run parse + verify inside a subprocess, put result into *queue*."""
    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import (ExprExtractionConfig, LatexExtractionConfig,
                                 parse, verify)

        j_with_env = f'${ref}$'
        gold_parsed = parse(
            j_with_env,
            extraction_mode='first_match',
            extraction_config=[
                LatexExtractionConfig(),
                ExprExtractionConfig(),
            ],
        )

        if len(gold_parsed) == 0:
            queue.put(('empty', ))
            return

        answer_parsed = parse(
            pred,
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
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode='first_match',
        )

        answer_correct = float(verify(answer_parsed, gold_parsed))
        queue.put(('ok', answer_correct, str(answer_parsed), str(gold_parsed)))
    except Exception as e:
        queue.put(('error', str(e)))


def _verify_with_timeout(pred, ref, timeout=10):
    """Wrap ``_verify_worker`` with a subprocess-level timeout."""
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_verify_worker, args=(pred, ref, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return {'timed_out': True}

    if queue.empty():
        return {'error': True}

    result = queue.get_nowait()
    if result[0] == 'empty':
        return {'skipped': True}
    elif result[0] == 'error':
        return {'error': True, 'msg': result[1]}
    else:
        return {
            'answer_correct': result[1],
            'answer_parsed': result[2],
            'gold_parsed': result[3],
        }


@ICL_EVALUATORS.register_module()
class MATHVerifyEvaluator(BaseEvaluator):

    def _score_single_prediction(self, prediction, reference):
        result = _verify_with_timeout(prediction, reference, timeout=10)

        if result.get('timed_out') or result.get('error'):
            detail = {
                'pred':
                prediction if result.get('timed_out') else result.get(
                    'msg', ''),
                'answer':
                reference,
                'correct':
                False,
            }
            return 0.0, detail

        if result.get('skipped'):
            return 0.0, {
                'pred': prediction,
                'answer': reference,
                'correct': False,
                'skipped': True,
            }

        answer_correct = result['answer_correct']
        detail = {
            'pred': result['answer_parsed'],
            'answer': result['gold_parsed'],
            'correct': True if answer_correct else False,
        }
        return answer_correct, detail

    def score(self, predictions, references, test_set=None):
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
            if isinstance(i, (list, tuple)):
                candidate_scores = []
                candidate_details = []
                for prediction in i:
                    score, detail = self._score_single_prediction(
                        prediction, j)
                    candidate_scores.append(score)
                    if detail is not None:
                        candidate_details.append(detail)

                answer_correct = (sum(candidate_scores) / len(candidate_scores)
                                  if candidate_scores else 0.0)
                correct += answer_correct
                details.append({
                    'pred': [detail['pred'] for detail in candidate_details],
                    'answer':
                    candidate_details[0]['answer'] if candidate_details else j,
                    'correct':
                    answer_correct == 1.0,
                    'pass_at_1':
                    answer_correct,
                    'candidate_details':
                    candidate_details,
                })
                continue

            answer_correct, detail = self._score_single_prediction(i, j)
            if detail is None:
                continue

            correct += answer_correct
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
